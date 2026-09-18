import asyncio
import logging
import timeit
from typing import List, Optional, Tuple, Union

import networkx as nx
import numpy as np
from google import genai
from google.genai.types import EmbedContentConfig

from ascent_domain.omop.models.uncertainty import UncertaintyResult
from ascent_domain.omop.models.uncertainty.schur_complement_entropy import SchurComplementEntropy
from ascent_domain.omop.models.uncertainty.uncertainty_kernel_laplacian import KernelLaplacianEntropy
from ascent_platform.config.runtime import settings
from ascent_platform.llm.factory import create_assistant
from ascent_platform.llm.http_options import gemini_http_options

logger = logging.getLogger(__name__)


class UncertaintyDecomposition:
    def __init__(self, t=10.0, norm_laplacian=True, similarity_method="prompt", use_schur_complement=True, gamma: float = 1.0):
        """
        Initialize Uncertainty Decomposition framework

        Args:
            t: Heat kernel diffusion parameter (default 10.0)
                - t=10.0: Optimal for normalized Laplacian with ~12 nodes
                - Ensures complete graphs give H≈0 (< 0.01 bits) while maintaining sensitivity
                - With normalized Laplacian: eigenvalues bounded in [0,2], stable across graph sizes
            norm_laplacian: Whether to use normalized Laplacian (default True, RECOMMENDED)
                - True: Eigenvalues always in [0,2] regardless of graph size (stable, recommended)
                - False: Eigenvalues scale with graph size (requires larger t)
            similarity_method: Method for computing similarities
            use_schur_complement: If True, use Schur complement method for H(R|I) (RECOMMENDED)
            gamma: Input kernel sharpening parameter for W_II matrix (default 1.0 = no sharpening).
                   Values > 1 suppress off-diagonal noise while preserving diagonal (1.0).
                   Recommended range: [1, 5, 10, 20]. Empirically γ=5 is optimal for clinical Text-to-SQL data.
                   This is the "Input Kernel Sharpening" (Mechanism A) for improving ESM sensitivity.
        """
        self.t = t
        self.norm_laplacian = norm_laplacian
        self.similarity_method = similarity_method
        self.use_schur_complement = use_schur_complement
        self.gamma = gamma
        self.kle = KernelLaplacianEntropy(t=t, norm_lapl=norm_laplacian, gamma=gamma)
        self.schur = SchurComplementEntropy(self.kle) if use_schur_complement else None

        if gamma != 1.0:
            logger.info(f"Input Kernel Sharpening ENABLED: γ={gamma} (will be applied to W_II matrix)")

    def compute_entropies(self, w_full, n_i, n_r):
        """
        Compute entropies using either Schur complement (RECOMMENDED) or subtraction method.

        SCHUR COMPLEMENT METHOD (use_schur_complement=True):
        Uses the Schur complement of the block matrix to DIRECTLY compute H(R|I),
        which GUARANTEES non-negativity. This is the theoretically principled approach.

        SUBTRACTION METHOD (use_schur_complement=False):
        Computes H(I,R) independently and subtracts H(I), which can give negative results
        due to numerical issues with spectral correction.

        Args:
            w_full: Full similarity matrix (after spectral correction)
            n_i: Number of interpretations
            n_r: Number of results

        Returns:
            tuple: (h_i, h_r, h_qir, h_r_given_i, h_r_given_i_naive)
                - h_i: H(I), interpretation entropy
                - h_r: H(R), result entropy
                - h_qir: H(I,R), joint entropy (via chain rule)
                - h_r_given_i: H(R|I), conditional entropy (Schur complement, guaranteed >= 0)
                - h_r_given_i_naive: H(R|I), naive subtraction (may be negative!)
        """
        logger.info("Using Schur complement method for conditional entropy computation")
        return self.schur.compute_entropies_schur(w_full, n_i, n_r)

    async def compute_decomposed_uncertainty(
        self,
        interpretations: List[str],
        results: List[str],
        mapping: List[int],
        assistant_type: Optional[str] = None,
        model_name: Optional[str] = None,
        embeddings_interp: Optional[np.ndarray] = None,
        embeddings_results: Optional[np.ndarray] = None,
        t: Optional[float] = None,
        max_concurrency: int = 25,
        interpretation_similarity_prompt: Optional[str] = None,
    ):
        """
        Computes decomposed uncertainty, running the two main similarity
        matrix calculations in parallel.

        Note: In FAST MODE, results will be placeholder strings. This still allows
        H(I) computation from interpretations, while H(R) and H(R|I) will be computed
        from the placeholder strings (which will be meaningless but won't crash).

        Args:
            interpretations: List of interpretation texts
            results: List of result texts
            mapping: Mapping from results to interpretations
            assistant_type: Assistant type for similarity computation
            model_name: Model name for the assistant
            embeddings_interp: Pre-computed embeddings for interpretations
            embeddings_results: Pre-computed embeddings for results
            t: Heat kernel diffusion parameter
            max_concurrency: Maximum concurrent API calls
            interpretation_similarity_prompt: Optional custom prompt for interpretation similarity.
                                             If provided, used only for W_II matrix computation.
        """
        if t is None:
            t = self.t

        logger.info("Computing input and output similarity matrices in parallel...")

        start_time = timeit.default_timer()  # Start timer

        # 1. Create two tasks. Each will eventually call the function above.
        # We pass the max_concurrency parameter down the chain.
        # shared_assistant = create_assistant(assistant_type=assistant_type, model_name=model_name)
        logger.info("Created shared assistant for parallel similarity computation.")
        logger.info("Computing input and output similarity matrices in parallel...")

        # Use custom prompt for interpretation similarity (W_II) if provided
        task_ii = self.compute_similarity_matrix(
            interpretations,
            self.similarity_method,
            embeddings=embeddings_interp,
            assistant_type=assistant_type,
            model_name=model_name,
            max_concurrency=max_concurrency,
            custom_entailment_prompt=interpretation_similarity_prompt,
        )
        # Results (W_RR) always use the default prompt
        task_rr = self.compute_similarity_matrix(
            results,
            self.similarity_method,
            embeddings=embeddings_results,
            assistant_type=assistant_type,
            model_name=model_name,
            max_concurrency=max_concurrency,
        )

        # 2. Run both tasks concurrently and get the results.
        (w_ii, g_ii), (w_rr, g_rr) = await asyncio.gather(task_ii, task_rr)

        # 3. Apply Input Kernel Sharpening (Mechanism A) to W_II if gamma != 1.0
        # This suppresses off-diagonal noise while preserving diagonal, improving
        # ESM sensitivity on fine-grained tasks (e.g., Clinical SQL, SituatedQA)
        w_ii_original = w_ii.copy()  # Keep original for logging/debugging
        if self.gamma != 1.0:
            logger.info(f"Applying Input Kernel Sharpening to W_II: γ={self.gamma}")
            w_ii = KernelLaplacianEntropy.sharpen_similarity_matrix(w_ii, self.gamma)
            # Log the effect
            if w_ii.shape[0] > 1:
                mask = ~np.eye(w_ii.shape[0], dtype=bool)
                mean_before = np.mean(w_ii_original[mask])
                mean_after = np.mean(w_ii[mask])
                logger.info(f"  Mean off-diagonal similarity: {mean_before:.4f} → {mean_after:.4f}")

        # The rest of the function continues sequentially as before, which is fast.
        logger.info("Computing input-output mappings...")
        w_ir = self.create_mapping_matrix(len(interpretations), len(results), mapping)
        w_ri = w_ir.T
        w_full = self.construct_full_matrix(w_ii, w_rr, w_ir, w_ri)

        logger.info("Computing uncertainties...")
        h_i, h_r, h_qir, h_r_given_i, h_r_given_i_naive = self.compute_entropies(w_full, len(interpretations), len(results))

        # Store the raw value for debugging
        h_r_given_i_raw = h_r_given_i

        # Apply non-negativity constraint only if needed
        if h_r_given_i < 0:
            logger.warning(
                f"Detected negative conditional entropy even with consistent calculation: "
                f"H_QIR ({h_qir:.6f}) < H_I ({h_i:.6f}). "
                f"This may indicate numerical issues. Enforcing non-negativity constraint."
            )
            h_r_given_i = 0.0

        # 7. Compute uncertainty contributions
        if h_qir > 0:
            # If we had to enforce non-negativity, adjust the contributions
            if h_r_given_i_raw < 0:
                ambiguity_contribution = 1.0
                result_contribution = 0.0
            else:
                ambiguity_contribution = h_i / h_qir
                result_contribution = h_r_given_i / h_qir
        else:
            ambiguity_contribution = np.nan
            result_contribution = np.nan
            logger.warning("Total uncertainty (H_QIR) is zero, using NaN for contributions")

        # 8. Create full uncertainty result with regime classification
        entropies_dict = {
            "H_I": h_i,
            "H_R": h_r,
            "H_QIR": h_qir,
            "H(R|I)_raw": h_r_given_i_raw,
            "H(R|I)": h_r_given_i,
            "H(R|I)_naive": h_r_given_i_naive,
        }
        uncertainty_result = UncertaintyResult.from_metrics(entropies_dict)

        elapsed_time = timeit.default_timer() - start_time
        logger.info(f"Total computation time: {elapsed_time:.2f} seconds")
        logger.info(f"Regime: {uncertainty_result.regime.value} → {uncertainty_result.recommended_action}")

        return {
            "matrices": {
                "W_II": w_ii,  # Sharpened if gamma != 1.0
                "W_II_original": w_ii_original,  # Always original (pre-sharpening)
                "W_RR": w_rr,
                "W_IR": w_ir,
                "W_RI": w_ri,
                "W_full": w_full,
            },
            "entropies": entropies_dict,
            "contributions": {"ambiguity": ambiguity_contribution, "result": result_contribution},
            "uncertainty_result": uncertainty_result,
            "graphs": {"G_II": g_ii, "G_RR": g_rr},
            "parameters": {"gamma": self.gamma, "t": t},
        }

    @staticmethod
    def get_google_embeddings(input_text: Union[str, List[str]], model: str = "gemini-embedding-exp-03-07") -> np.ndarray:
        """
        Generate embeddings for the provided text(s) using Google's Vertex AI.

        Args:
            input_text: A single string or list of strings to generate embeddings for
            model: The embedding model to use

        Returns:
            A numpy array of shape (n, embedding_dim) where n is the number of input texts
        """
        api_key = settings.GEMINI_API_KEY
        client = genai.Client(api_key=api_key, http_options=gemini_http_options())

        response = client.models.embed_content(
            model=model,
            contents=input_text,
            config=EmbedContentConfig(
                task_type="SEMANTIC_SIMILARITY",
            ),
        )

        embeddings_np = np.array([embedding.values for embedding in response.embeddings])

        return embeddings_np

    async def compute_similarity_matrix(
        self,
        texts: List[str],
        similarity_method: str,
        assistant_type: Optional[str] = None,
        model_name: Optional[str] = None,
        embeddings: Optional[np.ndarray] = None,
        max_concurrency: int = 25,
        custom_entailment_prompt: Optional[str] = None,
    ) -> Tuple[np.ndarray, nx.Graph]:
        """Compute similarity matrix for a set of texts.

        Args:
            texts: List of text strings to compare
            similarity_method: Method for computing similarities ('prompt', 'embedding', 'deberta')
            assistant_type: Assistant type for prompt-based similarity
            model_name: Model name for the assistant
            embeddings: Pre-computed embeddings (for 'embedding' method)
            max_concurrency: Maximum concurrent API calls
            custom_entailment_prompt: Optional custom prompt template for entailment checking
                                     (only used with 'prompt' similarity_method)
        """
        if similarity_method == "embedding":
            if embeddings is None:
                embeddings = self.get_google_embeddings(texts)
            graph = self.kle.get_embed_similarity_graph(texts, embeddings)
        elif similarity_method == "deberta":
            graph = self.kle.get_deberta_similarity_graph(texts)
        elif similarity_method == "prompt":
            assistant = create_assistant(assistant_type=assistant_type, model_name=model_name)
            logger.info(f"Compute similarity matrix: creating {assistant_type} assistant with model {model_name}")

            graph = await self.kle.get_prompt_similarity_graph(
                contents=texts, assistant=assistant, max_concurrency=max_concurrency, custom_prompt=custom_entailment_prompt
            )
        else:
            raise ValueError(f"Unknown similarity method: {similarity_method}")

        # Extract similarity matrix from graph (this logic is unchanged)
        n = len(texts)
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if graph.has_edge(i, j):
                    similarity_matrix[i, j] = graph[i][j]["weight"]
                else:
                    similarity_matrix[i, j] = 0.0

        return similarity_matrix, graph

    @staticmethod
    def create_mapping_matrix(n_interp, n_results, mapping):
        """
        Create binary mapping matrix W_IR

        Args:
            n_interp: Number of interpretations
            n_results: Number of results
            mapping: List where mapping[result_idx] = interpretation_idx
                     i.e., result i came from interpretation mapping[i]

        Returns:
            w_ir: n_interp × n_results matrix where w_ir[i, r] = 1 if interpretation i produced result r
        """
        w_ir = np.zeros((n_interp, n_results))
        for result_idx, interp_idx in enumerate(mapping):
            w_ir[interp_idx, result_idx] = 1.0
        return w_ir

    def construct_full_matrix(self, w_ii, w_rr, w_ir, w_ri):
        """
        Construct the full block similarity matrix W (Eq. 1 from paper).

        W = [W_II  W_IR]
            [W_RI  W_RR]

        No regularization is applied here — the paper only regularizes W_II
        for Schur complement inversion (ε·I in compute_schur_complement).
        """
        n_interp = w_ii.shape[0]
        n_results = w_rr.shape[0]

        # Construct initial full matrix
        w_full = np.zeros((n_interp + n_results, n_interp + n_results))
        w_full[:n_interp, :n_interp] = w_ii
        w_full[n_interp:, n_interp:] = w_rr
        w_full[:n_interp, n_interp:] = w_ir
        w_full[n_interp:, :n_interp] = w_ri

        # No full-matrix regularization — the paper only regularizes W_II
        # for Schur complement inversion (ε·I added in compute_schur_complement).
        # Heat kernels are robust to small negative eigenvalues.

        # Log eigenvalues for diagnostics
        eigenvalues = np.linalg.eigvalsh(w_full)
        min_eig = np.min(eigenvalues)
        max_eig = np.max(eigenvalues)
        logger.info(f"W_FULL eigenvalue range: [{min_eig:.6f}, {max_eig:.6f}]")

        if min_eig < -0.01:
            logger.info(f"W_FULL has negative eigenvalues (min={min_eig:.6f}). Heat kernel is robust to small negative eigenvalues.")

        return w_full
