import asyncio
import logging
from typing import Any, List

import networkx as nx
import numpy as np
import scipy.linalg
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# torch and transformers are imported lazily, inside the three deberta methods
# that need them. They are the heaviest dependencies in the tree (~408 MB) and
# are reachable ONLY via similarity_method="deberta"; production uses the
# default "prompt" method, which computes entailment through the LLM. Importing
# them at module scope pulled all of that into every container start.
def _require_torch():
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError("similarity_method='deberta' needs the optional 'deberta' extra: uv sync --extra deberta") from exc
    return torch, AutoModelForSequenceClassification, AutoTokenizer


class KernelLaplacianEntropy:
    def __init__(self, t=None, norm_lapl=True, similarity_method="prompt", gamma: float = 1.0):
        """
        Initialize KLE with parameters

        Args:
            t: Heat kernel diffusion parameter (if None, will be set based on similarity_method)
            norm_lapl: Whether to use normalized Laplacian (default True, RECOMMENDED)
            similarity_method: Default similarity method to determine t if not provided
            gamma: Input kernel sharpening parameter (default 1.0 = no sharpening).
                   Values > 1 suppress off-diagonal noise while preserving diagonal (1.0).
                   Recommended range: [1, 5, 10, 20]. Empirically γ=5 is optimal for clinical Text-to-SQL data.
        """
        # Set default t based on similarity method if not provided
        if t is None:
            if similarity_method == "embedding":
                t = 0.02
            elif similarity_method == "deberta":
                t = 10.0  # Optimal for normalized Laplacian: ensures complete graphs → H≈0 (< 0.01 bits)
            elif similarity_method == "prompt":
                t = 10.0  # Optimal for normalized Laplacian: ensures complete graphs → H≈0 (< 0.01 bits)
            else:  # unknown method
                logger.warning(f"Unknown similarity method: {similarity_method}. Using default 'prompt' method instead.")
                similarity_method = "prompt"
                t = 10.0  # Optimal for normalized Laplacian: ensures complete graphs → H≈0 (< 0.01 bits)

        self.t = t
        self.norm_lapl = norm_lapl
        self.gamma = gamma
        self.deberta_model = None
        self.deberta_tokenizer = None

        if gamma < 1.0:
            logger.warning(f"gamma={gamma} < 1.0 may cause unexpected behavior. Using gamma=1.0 instead.")
            self.gamma = 1.0

        logger.debug(f"Initialized KLE with t={self.t}, gamma={self.gamma} for {similarity_method} similarity method")

    @staticmethod
    def sharpen_similarity_matrix(matrix: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Apply element-wise power transformation to sharpen a similarity matrix.

        This is the "Input Kernel Sharpening" mechanism (Mechanism A) for improving
        ESM sensitivity on fine-grained tasks (e.g., SituatedQA, Clinical SQL).

        Mathematical Operation:
            W_new[i,j] = W_old[i,j] ** gamma

        Rationale:
            Standard embedding models are "low-frequency" observers that assign high
            similarity (>0.9) to inputs that are topically related but logically distinct
            (e.g., "Patient in 2020" vs. "Patient in 2021"). This causes the spectral
            decomposition to treat distinct constraints as identical, masking model instability.

            Raising similarity scores s ∈ [0,1] to a power γ > 1 suppresses off-diagonal
            noise while preserving the diagonal (1.0). This effectively increases the
            "spatial resolution" of the manifold, forcing the metric to respect fine-grained
            constraints.

        Args:
            matrix: Similarity matrix with values in [0, 1]. Diagonal should be 1.0.
            gamma: Sharpening exponent (default 1.0 = no sharpening).
                   Recommended range: [1, 5, 10, 20].
                   Empirically, γ ≈ 10 is optimal for temporal/clinical data.

        Returns:
            Sharpened similarity matrix with same shape.

        Example:
            >>> W = np.array([[1.0, 0.9], [0.9, 1.0]])
            >>> sharpen_similarity_matrix(W, gamma=10)
            array([[1.0, 0.349], [0.349, 1.0]])  # 0.9^10 ≈ 0.349
        """
        if gamma == 1.0:
            return matrix.copy()

        if gamma < 1.0:
            logger.warning(f"gamma={gamma} < 1.0 will amplify off-diagonal values. This is unusual.")

        # Element-wise power transformation
        # Note: This preserves diagonal=1.0 since 1.0^gamma = 1.0
        sharpened = np.power(matrix, gamma)

        # Ensure numerical stability - force diagonal to exactly 1.0
        np.fill_diagonal(sharpened, 1.0)

        # Log the sharpening effect
        if matrix.shape[0] > 1:
            # Calculate mean off-diagonal before and after
            mask = ~np.eye(matrix.shape[0], dtype=bool)
            mean_before = np.mean(matrix[mask])
            mean_after = np.mean(sharpened[mask])
            logger.debug(f"Kernel sharpening (γ={gamma}): mean off-diagonal {mean_before:.4f} → {mean_after:.4f}")

        return sharpened

    def compute_entropy_from_matrix(self, matrix):
        """Compute von Neumann entropy from similarity matrix"""
        # Create graph from similarity matrix
        G = nx.Graph()
        n = matrix.shape[0]
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(n):
                if matrix[i, j] > 0:
                    G.add_edge(i, j, weight=matrix[i, j])

        # Compute heat kernel
        kernel = self.heat_kernel(G)

        # Calculate von Neumann entropy
        entropy = self.vn_entropy(kernel)

        return entropy

    def load_deberta_model(self):
        """Load DeBERTa model for entailment checking"""
        torch, AutoModelForSequenceClassification, AutoTokenizer = _require_torch()
        if self.deberta_model is None:
            model_name = "microsoft/deberta-v2-xlarge-mnli"
            logging.info("Loading DeBerta model...")
            self.deberta_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.deberta_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.deberta_model.eval()

            # Move to GPU if available
            if torch.cuda.is_available():
                self.deberta_model = self.deberta_model.cuda()

        # This method is inside your KernelLaplacianEntropy class

    async def get_prompt_similarity_graph(
        self, contents: List[str], assistant: Any, max_concurrency: int = 25, custom_prompt: str = None
    ) -> nx.Graph:
        """
        Constructs a graph based on prompt-based entailment scores, with all
        API calls running in parallel, controlled by a semaphore to respect rate limits.

        Args:
            contents: List of text strings to compare
            assistant: The LLM assistant to use
            max_concurrency: Maximum number of concurrent API calls
            custom_prompt: Optional custom prompt template with {text1} and {text2} placeholders
        """
        nodes = range(len(contents))

        # 1. Create a semaphore using the provided concurrency limit.
        semaphore = asyncio.Semaphore(max_concurrency)

        # 2. Define a helper function to wrap a single API call with the semaphore.
        async def get_entailment_with_semaphore(text1: str, text2: str) -> float:
            async with semaphore:
                result = await self.check_entailment_prompt(text1, text2, assistant, custom_prompt)
                return result

        # 3. Create a list of all tasks that need to be run.
        tasks = []
        edge_indices = []
        for i in range(len(contents)):
            for j in range(i + 1, len(contents)):
                tasks.append(get_entailment_with_semaphore(contents[i], contents[j]))
                tasks.append(get_entailment_with_semaphore(contents[j], contents[i]))
                edge_indices.append((i, j))

        # 4. Run all tasks concurrently.
        all_weights = await asyncio.gather(*tasks, return_exceptions=True)

        # 5. Process the results and build the edges.
        edges = []
        for i in range(len(contents)):
            edges.append((i, i, 1.0))

        for idx, (i, j) in enumerate(edge_indices):
            weight_1to2 = all_weights[idx * 2]
            weight_2to1 = all_weights[idx * 2 + 1]

            if isinstance(weight_1to2, Exception) or isinstance(weight_2to1, Exception):
                logger.error(f"Failed to calculate similarity for pair ({i}, {j}). Skipping edge.")
                continue

            combined_weight = (weight_1to2 + weight_2to1) / 2
            if combined_weight > 0:
                edges.append((i, j, combined_weight))

        # Create and return the graph
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from(edges)
        return G

    @staticmethod
    async def check_entailment_prompt(text1, text2, assistant, custom_prompt: str = None):
        """
        Check entailment using prompt-based approach with numerical awareness

        Args:
            text1: First text
            text2: Second text
            assistant: The LLM assistant to use for entailment checking
            custom_prompt: Optional custom prompt template with {text1} and {text2} placeholders.
                          If None, uses the default prompt.

        Returns:
            float: Entailment score between 0 and 1
        """
        if custom_prompt:
            # Use custom prompt template
            prompt = custom_prompt.format(text1=text1, text2=text2)
        else:
            # Use default prompt
            prompt = f"""
        Task: Determine if the first statement entails (implies or contains the same information as) the second statement.

        Pay special attention to numerical values in both statements:
        - If both statements contain the same numbers, they are more likely to entail each other
        - If the statements contain different numbers, the degree of difference should be reflected in your score
        - Small numerical differences (within 5%) should have minimal impact on the score
        - Large numerical differences should significantly reduce the entailment score

        Statement 1: "{text1}"
        Statement 2: "{text2}"

        Analyze both statements carefully, focusing on both the semantic meaning and any numerical values.

        Return a single number between 0 and 1 representing the entailment score:
        - 1.0: Perfect entailment (Statement 1 completely entails Statement 2)
        - 0.0: No entailment (Statement 1 contradicts or is unrelated to Statement 2)
        - Values between 0 and 1 indicate partial entailment

        Your response should be ONLY a number between 0 and 1, with no additional text.
        """

        try:
            # Get structured response using the schema
            response = await assistant.get_response(prompt=prompt, response_schema=EntailmentScore, quiet=True)

            if not isinstance(response, EntailmentScore):
                error_msg = f"Failed to generate properly structured entailment score for statements: '{text1}' and '{text2}'"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Extract the score
            return response.score

        except Exception as e:
            error_msg = f"Error in prompt-based entailment check: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    @staticmethod
    def get_embed_similarity_graph(contents, embeddings):
        """
        Construct a graph based on embedding cosine similarity with self-loops
        """
        # Normalize embeddings
        normalized_embeddings = [emb / np.linalg.norm(emb) for emb in embeddings]

        # Create nodes and edges
        nodes = range(len(contents))
        edges = []

        # Add self-loops with weight 1.0
        for i in range(len(contents)):
            edges.append((i, i, 1.0))  # Add self-loop with maximum similarity

        # Calculate pairwise similarities (symmetric)
        for i in range(len(contents)):
            for j in range(i + 1, len(contents)):
                # Calculate cosine similarity
                similarity = np.dot(normalized_embeddings[i], normalized_embeddings[j])

                # Add edge if similarity is positive
                if similarity > 0:
                    edges.append((i, j, similarity))

        # Create graph
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from(edges)

        return G

    def get_deberta_similarity_graph(self, contents):
        """
        Construct a graph based on DeBERTa entailment scores

        Args:
            contents: List of text strings

        Returns:
            networkx.Graph: Graph with nodes for each text and weighted edges
        """
        # Load DeBERTa model if not already loaded
        self.load_deberta_model()

        # Create nodes and edges
        nodes = range(len(contents))
        edges = []

        for i, text1 in enumerate(contents):
            for j in range(i + 1, len(contents)):
                text2 = contents[j]

                # Check entailment in both directions
                weight_1to2 = self.check_entailment_deberta(text1, text2)
                weight_2to1 = self.check_entailment_deberta(text2, text1)

                # Combine weights (symmetric representation)
                combined_weight = weight_1to2 + weight_2to1

                # Add edge if there's some entailment
                if combined_weight > 0:
                    edges.append((i, j, combined_weight))

        # Create graph
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from(edges)

        return G

    def check_entailment_deberta(self, premise, hypothesis):
        """
        Check entailment using DeBERTa

        Args:
            premise: Premise text
            hypothesis: Hypothesis text

        Returns:
            float: Entailment score
        """
        torch, _, _ = _require_torch()
        # Tokenize
        inputs = self.deberta_tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Get model prediction
        with torch.no_grad():
            outputs = self.deberta_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)

        # Get entailment probability (index 2 is entailment in MNLI)
        entailment_prob = probs[0, 2].item()

        return entailment_prob

    def heat_kernel(self, G):
        """
        Compute heat kernel from graph using the paper's original approach

        Args:
            G: networkx.Graph

        Returns:
            numpy.ndarray: Heat kernel matrix
        """
        # Get Laplacian using NetworkX's built-in functions
        if isinstance(G, nx.DiGraph):
            L = nx.directed_laplacian_matrix(G)
        elif self.norm_lapl:
            L = nx.normalized_laplacian_matrix(G).toarray()
        else:
            L = nx.laplacian_matrix(G).toarray()

        # Compute heat kernel with error handling
        try:
            K = scipy.linalg.expm(-self.t * L)
        except Exception as e:
            # Fallback for numerical stability
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            K = eigenvectors @ np.diag(np.exp(-self.t * eigenvalues)) @ eigenvectors.T
            logger.warning(f"Used eigendecomposition fallback due to: {e}")

        return K

    def vn_entropy(self, kernel, scale=True, normalize=True, jitter=1e-10):
        """
        Compute von Neumann entropy of kernel matrix with better handling of edge cases
        """
        n = kernel.shape[0]

        # Check if kernel is all zeros or has NaNs
        if np.all(kernel == 0):
            raise ValueError("Invalid kernel matrix: all elements are zero")
        if np.isnan(kernel).any():
            raise ValueError("Invalid kernel matrix: contains NaN values")

        # Normalize kernel if requested
        if normalize:
            trace = np.trace(kernel)
            if trace <= 0:
                raise ValueError(f"Invalid kernel matrix: trace is non-positive ({trace})")
            kernel = kernel / trace

        # Add small jitter for numerical stability
        kernel = kernel + np.eye(n) * jitter

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(kernel)
        positive_eigenvalues = eigenvalues[eigenvalues > 0]

        if len(positive_eigenvalues) == 0:
            raise ValueError("Invalid kernel matrix: no positive eigenvalues found")

        # Compute entropy: -sum(λ_i * log(λ_i))
        entropy = -np.sum(eigenvalues * np.log(eigenvalues))

        # Scale if requested
        if scale:
            entropy = entropy / np.log(n)

        # Ensure entropy is within bounds
        if scale:
            entropy = max(0.0, min(1.0, entropy))  # Clamp between 0 and 1
        else:
            entropy = max(0.0, min(np.log(n), entropy))  # Clamp between 0 and log(n)

        return entropy


class EntailmentScore(BaseModel):
    score: float = Field(
        description="A number between 0 and 1 representing the entailment score",
        ge=0.0,  # greater than or equal to 0
        le=1.0,  # less than or equal to 1
    )

    @field_validator("score")
    def validate_score_range(cls, v):
        if v < 0.0 or v > 1.0:
            logger.warning(f"Entailment score {v} is outside valid range [0,1]. Clamping to valid range.")
            return max(0.0, min(1.0, v))
        return v
