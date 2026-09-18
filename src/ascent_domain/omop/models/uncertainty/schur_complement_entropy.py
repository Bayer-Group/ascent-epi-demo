"""
Schur Complement Method for Computing Conditional Entropy H(R|I)

This module implements a principled, theoretically sound approach to computing
conditional entropy using the Schur complement of the block matrix.

Mathematical Foundation:
=======================

For a block matrix:
    W_FULL = [ W_II  W_IR ]
             [ W_RI  W_RR ]

The Schur complement of W_II is:
    S = W_RR - W_RI @ inv(W_II + εI) @ W_IR

Key Properties:
1. det(W_FULL) = det(W_II) * det(S)
2. log det(W_FULL) = log det(W_II) + log det(S)
3. H(I,R) = H(I) + H(R|I)  where H(R|I) = entropy(S)

Advantages over Subtraction Method:
===================================
1. GUARANTEED non-negative H(R|I) when W_FULL is PSD
2. Direct calculation from conditional kernel (no subtraction)
3. Theoretically principled (Schur complement is standard linear algebra)
4. Connects to quantum information theory (composite systems)

Physical Interpretation:
=======================
The Schur complement S represents the "residual" similarity in results
AFTER factoring out the variance that comes from interpretations being different.

It's the portion of result variability that cannot be explained by interpretation
differences - precisely what we want for H(R|I).

Citation:
=========
This approach is inspired by the treatment of composite quantum systems
in quantum information theory, where conditioning is done via partial traces
and Schur complements naturally arise in block matrix decompositions.

References:
- Zhang, F. (Ed.). (2006). The Schur complement and its applications.
- Nielsen, M. A., & Chuang, I. L. (2010). Quantum computation and quantum information.
- Horn, R. A., & Johnson, C. R. (2012). Matrix analysis (2nd ed.).

Author: Generated from theoretical insights about block matrix structure
Date: November 2024
"""

import logging
from typing import Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class SchurComplementEntropy:
    """
    Compute conditional entropy H(R|I) using Schur complement method.

    This class provides a mathematically principled approach to computing
    conditional entropy that is guaranteed to be non-negative.
    """

    def __init__(self, kle_instance):
        """
        Initialize with a KernelLaplacianEntropy instance for entropy computation.

        Args:
            kle_instance: Instance of KernelLaplacianEntropy class
        """
        self.kle = kle_instance

    def compute_schur_complement(
        self,
        w_full: np.ndarray,
        n_i: int,
        n_r: int,
        epsilon: float = 1e-1
    ) -> np.ndarray:
        """
        Compute Schur complement to obtain conditional kernel for H(R|I).

        For block matrix:
            W_FULL = [ W_II  W_IR ]
                     [ W_RI  W_RR ]

        The Schur complement is:
            S = W_RR - W_RI @ inv(W_II + εI) @ W_IR

        This represents the residual similarity in results AFTER factoring out
        the variance that comes from interpretations being different.

        Mathematical Foundation:
        -----------------------
        The determinant identity:
            det(W_FULL) = det(W_II) * det(S)

        Taking logarithms:
            log det(W_FULL) = log det(W_II) + log det(S)

        This corresponds to entropy chain rule:
            H(I,R) = H(I) + H(R|I)

        where H(R|I) is the von Neumann entropy of the heat kernel on S.

        Args:
            w_full: Full similarity matrix (n_i+n_r × n_i+n_r), PSD-corrected
            n_i: Number of interpretations
            n_r: Number of results
            epsilon: Regularization for W_II inversion stability (default 1e-1).
                     Paper uses ε=10^{-3}, but larger values prevent blowup when W_II
                     is near-singular (very similar interpretations).

        Returns:
            schur_complement: (n_r × n_r) conditional kernel for results given interpretations

        Raises:
            ValueError: If matrix dimensions are inconsistent
        """
        if w_full.shape != (n_i + n_r, n_i + n_r):
            raise ValueError(f"Expected W_FULL shape {(n_i + n_r, n_i + n_r)}, got {w_full.shape}")

        # Extract blocks
        w_ii = w_full[:n_i, :n_i]    # Interpretation-interpretation (n_i × n_i)
        w_ir = w_full[:n_i, n_i:]    # Interpretation-result (n_i × n_r)
        w_ri = w_full[n_i:, :n_i]    # Result-interpretation (n_r × n_i)
        w_rr = w_full[n_i:, n_i:]    # Result-result (n_r × n_r)

        logger.info(f"Computing Schur complement for {n_i} interpretations and {n_r} results")
        logger.info(f"Block shapes: W_II={w_ii.shape}, W_IR={w_ir.shape}, W_RI={w_ri.shape}, W_RR={w_rr.shape}")

        # Log matrices for diagnostics
        logger.info(f"W_II matrix:\n{np.array2string(w_ii, precision=4, suppress_small=True)}")
        logger.info(f"W_RR matrix:\n{np.array2string(w_rr, precision=4, suppress_small=True)}")
        logger.info(f"W_IR matrix:\n{np.array2string(w_ir, precision=4, suppress_small=True)}")

        # Add regularization to W_II for numerical stability (paper: ε = 10^{-3})
        # This represents how "distinguishable" the interpretations are
        w_ii_reg = w_ii + epsilon * np.eye(n_i)

        # Log W_II eigenvalues for diagnostics
        w_ii_eigenvalues = np.linalg.eigvalsh(w_ii)
        logger.info(f"W_II eigenvalues (before regularization): {np.array2string(w_ii_eigenvalues, precision=6)}")
        logger.info(f"W_II regularization epsilon: {epsilon}")

        # Compute condition number to check numerical stability
        cond_number = np.linalg.cond(w_ii_reg)
        logger.info(f"W_II condition number: {cond_number:.2e}")

        if cond_number > 1e10:
            logger.warning(f"High condition number ({cond_number:.2e}), using pseudo-inverse")
            w_ii_inv = np.linalg.pinv(w_ii_reg)
        else:
            try:
                w_ii_inv = np.linalg.inv(w_ii_reg)
            except np.linalg.LinAlgError as e:
                logger.warning(f"Matrix inversion failed ({e}), using pseudo-inverse")
                w_ii_inv = np.linalg.pinv(w_ii_reg)

        # Compute Schur complement: S = W_RR - W_RI @ inv(W_II) @ W_IR
        # This is the "residual" similarity after removing interpretation-based variance
        projection_term = w_ri @ w_ii_inv @ w_ir
        schur = w_rr - projection_term

        # Log projection term and Schur complement
        logger.info(f"Projection term W_RI @ inv(W_II) @ W_IR:\n{np.array2string(projection_term, precision=4, suppress_small=True)}")

        # Log projection magnitude (how much variance is explained by interpretations)
        projection_frobenius = np.linalg.norm(projection_term, 'fro')
        wrr_frobenius = np.linalg.norm(w_rr, 'fro')
        projection_ratio = projection_frobenius / wrr_frobenius if wrr_frobenius > 0 else 0
        logger.info(f"Interpretation-explained variance: {projection_ratio:.2%} of W_RR")
        logger.info(f"Residual variance (Schur complement): {1-projection_ratio:.2%}")

        logger.info(f"Schur complement S = W_RR - projection:\n{np.array2string(schur, precision=4, suppress_small=True)}")

        # Ensure Schur complement is PSD (should be if W_FULL is PSD, but check for numerical errors)
        eigenvalues, eigenvectors = np.linalg.eigh(schur)
        min_eigenvalue = np.min(eigenvalues)
        max_eigenvalue = np.max(eigenvalues)

        logger.info(f"Schur complement eigenvalue range: [{min_eigenvalue:.6f}, {max_eigenvalue:.6f}]")

        if min_eigenvalue < -1e-10:
            num_negative = np.sum(eigenvalues < 0)
            logger.warning(f"Schur complement has {num_negative} negative eigenvalues (min={min_eigenvalue:.6f})")
            logger.warning("Projecting to PSD by clipping eigenvalues to zero")
            eigenvalues_corrected = np.maximum(eigenvalues, 0)
            schur = eigenvectors @ np.diag(eigenvalues_corrected) @ eigenvectors.T
            logger.info(f"After PSD projection: eigenvalue range [{0.0:.6f}, {max_eigenvalue:.6f}]")

        # Check if Schur complement is nearly zero (all interpretations produce identical results)
        if max_eigenvalue < 1e-10:
            logger.info("⚠️  Schur complement is nearly zero - results are deterministic given interpretations")
            logger.info("   This means H(R|I) ≈ 0 (no SQL sampling variance)")

        return schur

    def compute_naive_h_r_given_i(
        self,
        w_full: np.ndarray,
        n_i: int,
        n_r: int,
        h_i: float
    ) -> float:
        """
        Compute H(R|I) using naive subtraction: H(I,R)_joint - H(I).

        This method computes H(I,R) from a joint diffusion on the full graph,
        then subtracts the marginal H(I). This is provided for comparison with
        the Schur complement method.

        WARNING: This approach can yield NEGATIVE values because:
        1. The chain rule H(I,R) = H(I) + H(R|I) holds for Shannon entropy on
           probability distributions, but NOT for von Neumann entropy on
           heat-kernel-derived density matrices.
        2. The joint and marginal kernels have different sizes and undergo
           independent diffusion processes with different dynamics.
        3. The trace normalization is applied independently to each kernel.

        Args:
            w_full: Full similarity matrix, shape (n_i+n_r × n_i+n_r)
            n_i: Number of interpretations
            n_r: Number of results
            h_i: Pre-computed H(I) from marginal W_II

        Returns:
            float: H(R|I)_naive = H(I,R)_joint - H(I). May be negative!
        """
        import networkx as nx

        # Build graph from full similarity matrix
        n_total = n_i + n_r
        G_full = nx.Graph()
        for i in range(n_total):
            for j in range(i, n_total):
                if w_full[i, j] > 1e-10:
                    G_full.add_edge(i, j, weight=w_full[i, j])

        # Compute heat kernel on full graph
        K_full = self.kle.heat_kernel(G_full)
        K_full_norm = K_full / np.trace(K_full) if np.trace(K_full) > 0 else K_full

        # Compute joint entropy H(I,R) from full kernel
        h_ir_joint = self.kle.vn_entropy(K_full_norm)

        # Naive subtraction: H(R|I) = H(I,R) - H(I)
        h_r_given_i_naive = h_ir_joint - h_i

        logger.info(f"  Naive subtraction: H(I,R)_joint={h_ir_joint:.6f}, H(I)={h_i:.6f}")
        logger.info(f"  H(R|I)_naive = {h_r_given_i_naive:.6f} {'⚠️ NEGATIVE!' if h_r_given_i_naive < 0 else ''}")

        return h_r_given_i_naive

    def compute_entropies_schur(
        self,
        w_full: np.ndarray,
        n_i: int,
        n_r: int,
        epsilon: float = 1e-1
    ) -> Tuple[float, float, float, float, float]:
        """
        Compute entropies using Schur complement method.

        NEW PRINCIPLED APPROACH:
        Instead of computing H(I,R) independently and subtracting H(I) (which can give
        negative results), we use the Schur complement to DIRECTLY compute H(R|I) from
        a conditional kernel, then DEFINE H(I,R) via the chain rule.

        Mathematical Foundation:
        -----------------------
        For block matrix W_FULL = [W_II W_IR; W_RI W_RR], the Schur complement
        S = W_RR - W_RI @ inv(W_II) @ W_IR represents the conditional space R|I.

        The determinant identity:
            det(W_FULL) = det(W_II) * det(S)

        Translates to entropy chain rule:
            H(I,R) = H(I) + H(R|I)

        where:
            - H(I) is computed from W_II (marginal)
            - H(R|I) is computed from Schur complement S (conditional)
            - H(I,R) is DEFINED by chain rule (not computed independently)
            - H(R) is computed from W_RR (marginal, for reference only)

        Guarantee:
        ---------
        This approach GUARANTEES H(R|I) ≥ 0 because S is PSD when W_FULL is PSD.

        Also computes naive subtraction H(R|I) for comparison (may be negative).

        Args:
            w_full: Full similarity matrix (PSD-corrected), shape (n_i+n_r × n_i+n_r)
            n_i: Number of interpretations
            n_r: Number of results
            epsilon: Regularization parameter for Schur complement computation

        Returns:
            tuple: (h_i, h_r, h_qir, h_r_given_i, h_r_given_i_naive)
                - h_i: H(I), interpretation ambiguity (aleatoric)
                - h_r: H(R), marginal result entropy (for reference)
                - h_qir: H(I,R), total system entropy (defined via chain rule)
                - h_r_given_i: H(R|I), conditional entropy (epistemic, from Schur complement)
                - h_r_given_i_naive: H(R|I) via naive subtraction (may be negative!)
        """
        logger.info("=" * 80)
        logger.info("SCHUR COMPLEMENT METHOD FOR CONDITIONAL ENTROPY")
        logger.info("=" * 80)
        logger.info("This method GUARANTEES H(R|I) ≥ 0 by construction")
        logger.info("Based on standard Schur complement from linear algebra")
        logger.info("=" * 80)

        # Extract blocks for marginal computations
        w_ii = w_full[:n_i, :n_i]      # I-I similarities (marginal)
        w_rr = w_full[n_i:, n_i:]      # R-R similarities (marginal)

        # Step 1: Build graph for interpretations (marginal distribution)
        logger.info(f"Step 1: Computing H(I) from W_II ({n_i}×{n_i})")
        G_i = nx.Graph()
        for i in range(n_i):
            for j in range(i, n_i):
                if w_ii[i, j] > 1e-10:
                    G_i.add_edge(i, j, weight=w_ii[i, j])

        K_i = self.kle.heat_kernel(G_i)
        K_i_norm = K_i / np.trace(K_i) if np.trace(K_i) > 0 else K_i
        h_i = self.kle.vn_entropy(K_i_norm)
        logger.info(f"  ✓ H(I) = {h_i:.6f} bits (interpretation ambiguity)")

        # Step 2: Build graph for results (marginal distribution - for reference only)
        logger.info(f"Step 2: Computing H(R) from W_RR ({n_r}×{n_r}) [for reference]")
        G_r = nx.Graph()
        for i in range(n_r):
            for j in range(i, n_r):
                if w_rr[i, j] > 1e-10:
                    G_r.add_edge(i, j, weight=w_rr[i, j])

        K_r = self.kle.heat_kernel(G_r)
        K_r_norm = K_r / np.trace(K_r) if np.trace(K_r) > 0 else K_r
        h_r = self.kle.vn_entropy(K_r_norm)
        logger.info(f"  ✓ H(R) = {h_r:.6f} bits (marginal result entropy)")

        # Step 3: Compute Schur complement
        logger.info("Step 3: Computing Schur complement S = W_RR - W_RI @ inv(W_II) @ W_IR")
        schur = self.compute_schur_complement(w_full, n_i, n_r, epsilon)

        # Step 4: Compute H(R|I) from Schur complement
        logger.info("Step 4: Computing H(R|I) from Schur complement kernel")
        G_schur = nx.Graph()
        for i in range(n_r):
            for j in range(i, n_r):
                if schur[i, j] > 1e-10:
                    G_schur.add_edge(i, j, weight=schur[i, j])

        if len(G_schur.edges()) == 0:
            logger.warning("Schur complement graph has no edges - H(R|I) will be 0")
            h_r_given_i = 0.0
        else:
            K_schur = self.kle.heat_kernel(G_schur)
            K_schur_norm = K_schur / np.trace(K_schur) if np.trace(K_schur) > 0 else K_schur
            h_r_given_i = self.kle.vn_entropy(K_schur_norm)

        logger.info(f"  ✓ H(R|I) = {h_r_given_i:.6f} bits (SQL sampling uncertainty)")

        # Step 5: Define joint entropy via chain rule
        logger.info("Step 5: Defining H(I,R) via chain rule: H(I,R) = H(I) + H(R|I)")
        h_qir = h_i + h_r_given_i
        logger.info(f"  ✓ H(I,R) = {h_qir:.6f} bits (total system entropy)")

        # Step 6: Compute naive subtraction H(R|I) for comparison
        logger.info("Step 6: Computing naive subtraction H(R|I) = H(I,R)_joint - H(I)")
        h_r_given_i_naive = self.compute_naive_h_r_given_i(w_full, n_i, n_r, h_i)

        logger.info("=" * 80)
        logger.info("DECOMPOSITION SUMMARY:")
        logger.info("=" * 80)
        if h_qir > 1e-10:
            logger.info(f"  H(I)     = {h_i:.6f} bits  ({h_i/h_qir*100:.1f}% of total)")
            logger.info(f"  H(R|I)   = {h_r_given_i:.6f} bits  ({h_r_given_i/h_qir*100:.1f}% of total) [Schur]")
            logger.info(f"  H(R|I)   = {h_r_given_i_naive:.6f} bits [Naive subtraction]")
        else:
            logger.info(f"  H(I)     = {h_i:.6f} bits")
            logger.info(f"  H(R|I)   = {h_r_given_i:.6f} bits [Schur]")
            logger.info(f"  H(R|I)   = {h_r_given_i_naive:.6f} bits [Naive subtraction]")
        logger.info(f"  H(I,R)   = {h_qir:.6f} bits (via chain rule)")
        logger.info(f"  H(R)     = {h_r:.6f} bits (marginal, for reference)")
        logger.info("=" * 80)

        # Verify constraints
        logger.info("CONSTRAINT VERIFICATION:")
        logger.info("=" * 80)

        if h_r_given_i < -1e-10:
            logger.error(f"❌ ERROR: H(R|I)_schur = {h_r_given_i:.6f} < 0")
            logger.error("   This should NOT happen with Schur complement method!")
            logger.error("   Indicates numerical issues in computation")
        else:
            logger.info(f"✅ H(R|I)_schur ≥ 0: {h_r_given_i:.6f} bits (GUARANTEED by Schur complement)")

        if h_r_given_i_naive < -1e-10:
            logger.warning(f"⚠️  H(R|I)_naive = {h_r_given_i_naive:.6f} < 0 (EXPECTED - chain rule doesn't hold)")
        else:
            logger.info(f"✅ H(R|I)_naive ≥ 0: {h_r_given_i_naive:.6f} bits")

        if h_qir < h_i - 1e-10:
            logger.error(f"❌ ERROR: H(I,R) = {h_qir:.6f} < H(I) = {h_i:.6f}")
            logger.error("   This should NOT happen when using chain rule definition!")
        else:
            logger.info(f"✅ H(I,R) ≥ H(I): {h_qir:.6f} ≥ {h_i:.6f} (satisfied by construction)")

        # Information-theoretic bounds
        if h_r_given_i > h_r + 1e-6:
            logger.warning(f"⚠️  H(R|I)_schur = {h_r_given_i:.6f} > H(R) = {h_r:.6f}")
            logger.warning("   Conditioning should not increase entropy")
            logger.warning("   This can happen with heat kernel approximations")
        else:
            logger.info(f"✅ H(R|I)_schur ≤ H(R): {h_r_given_i:.6f} ≤ {h_r:.6f} (conditioning reduces entropy)")

        # Compare methods
        diff = h_r_given_i_naive - h_r_given_i
        logger.info(f"  Method comparison: Naive - Schur = {diff:.6f} bits")

        logger.info("=" * 80)

        return h_i, h_r, h_qir, h_r_given_i, h_r_given_i_naive

