"""
Uncertainty Configuration

Configuration dataclasses for CLUES uncertainty decomposition framework.
Contains default thresholds from the paper for regime classification.

Paper Reference:
    "Separating Input Ambiguity from Model Instability in Large Language Models:
     A Clinical Text-to-SQL Case Study" (LREC 2026)

Regime Definitions (Section 3.3):
    - Regime I (Confident): Low H_I, Low H(R|I) → Auto-answer
    - Regime II (Ambiguity): High H_I, Low H(R|I) → Clarification dialogue
    - Regime III (Instability): Low H_I, High H(R|I) → Human review
    - Regime IV (Compound): High H_I, High H(R|I) → Clarification + Review
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class UncertaintyRegime(str, Enum):
    """Uncertainty regime classification based on H_I and H(R|I) thresholds."""
    CONFIDENT = "confident"           # Regime I: Low ambiguity, low instability → auto-answer
    AMBIGUITY = "ambiguity"           # Regime II: High ambiguity, low instability → clarification
    INSTABILITY = "instability"       # Regime III: Low ambiguity, high instability → human review
    COMPOUND = "compound"             # Regime IV: High ambiguity, high instability → clarification + review


@dataclass
class UncertaintyThresholds:
    """
    Threshold configuration for CLUES uncertainty regime classification.

    Default values are median-based thresholds from paper experiments:
    - H_I threshold: Separates low/high interpretation ambiguity
    - H(R|I) threshold: Separates low/high model instability

    These thresholds were calibrated on:
    - AmbigQA/SituatedQA (open-domain QA)
    - Clinical Text-to-SQL benchmark

    For deployment, thresholds should be calibrated on your specific domain.
    """
    # Paper default thresholds (median-based from deployment study, Section 6.3)
    # These are reasonable starting points for clinical Text-to-SQL
    h_i_threshold: float = 0.2          # H(I) threshold for ambiguity (median from paper)
    h_r_given_i_threshold: float = 0.004  # H(R|I) threshold for instability (median from paper)

    # Heat kernel parameter (must match what was used in computation)
    t: float = 10.0

    # Gamma sharpening parameter (1.0 = no sharpening)
    gamma: float = 1.0

    def classify_regime(self, h_i: float, h_r_given_i: float) -> UncertaintyRegime:
        """
        Classify a query into one of four uncertainty regimes.

        Args:
            h_i: Interpretation entropy (ambiguity score)
            h_r_given_i: Conditional result entropy (instability score)

        Returns:
            UncertaintyRegime enum value
        """
        high_ambiguity = h_i >= self.h_i_threshold
        high_instability = h_r_given_i >= self.h_r_given_i_threshold

        if high_ambiguity and high_instability:
            return UncertaintyRegime.COMPOUND
        elif high_ambiguity:
            return UncertaintyRegime.AMBIGUITY
        elif high_instability:
            return UncertaintyRegime.INSTABILITY
        else:
            return UncertaintyRegime.CONFIDENT

    def get_recommended_action(self, regime: UncertaintyRegime) -> str:
        """
        Get the recommended action for a given uncertainty regime.

        Args:
            regime: UncertaintyRegime classification

        Returns:
            Human-readable recommendation string
        """
        actions = {
            UncertaintyRegime.CONFIDENT: "Auto-answer: High confidence, proceed with response.",
            UncertaintyRegime.AMBIGUITY: "Clarification needed: Question is ambiguous, ask user to clarify.",
            UncertaintyRegime.INSTABILITY: "Human review: Model unstable, flag for expert review.",
            UncertaintyRegime.COMPOUND: "Clarification + Review: Both ambiguous and unstable, requires intervention.",
        }
        return actions[regime]


@dataclass
class UncertaintyResult:
    """
    Complete uncertainty analysis result with decomposed metrics and regime classification.

    This is the main output structure for uncertainty analysis, containing:
    - Raw entropy metrics (H_I, H_R, H(R|I))
    - Regime classification
    - Recommended action
    """
    # Decomposed entropy metrics
    h_i: float                          # H(I): Interpretation entropy (ambiguity)
    h_r: float                          # H(R): Result entropy (total diversity)
    h_r_given_i: float                  # H(R|I): Conditional entropy (instability)
    h_joint: float                      # H(I,R): Joint entropy

    # Naive subtraction result (for comparison, may be negative)
    h_r_given_i_naive: Optional[float] = None

    # Regime classification
    regime: UncertaintyRegime = UncertaintyRegime.CONFIDENT
    recommended_action: str = ""

    # Contribution percentages
    ambiguity_contribution: float = 0.0
    instability_contribution: float = 0.0

    @classmethod
    def from_metrics(
        cls,
        entropies: dict,
        thresholds: Optional[UncertaintyThresholds] = None
    ) -> "UncertaintyResult":
        """
        Create UncertaintyResult from the entropies dict returned by UncertaintyDecomposition.

        Args:
            entropies: Dict with keys 'H_I', 'H_R', 'H(R|I)', 'H_QIR', 'H(R|I)_naive'
            thresholds: Optional threshold config (uses defaults if None)

        Returns:
            UncertaintyResult with all metrics and regime classification
        """
        if thresholds is None:
            thresholds = UncertaintyThresholds()

        h_i = entropies.get("H_I", 0.0)
        h_r = entropies.get("H_R", 0.0)
        h_r_given_i = entropies.get("H(R|I)", 0.0)
        h_joint = entropies.get("H_QIR", 0.0)
        h_r_given_i_naive = entropies.get("H(R|I)_naive")

        # Classify regime
        regime = thresholds.classify_regime(h_i, h_r_given_i)
        recommended_action = thresholds.get_recommended_action(regime)

        # Compute contributions
        if h_joint > 0:
            ambiguity_contribution = h_i / h_joint
            instability_contribution = h_r_given_i / h_joint
        else:
            ambiguity_contribution = 0.0
            instability_contribution = 0.0

        return cls(
            h_i=h_i,
            h_r=h_r,
            h_r_given_i=h_r_given_i,
            h_joint=h_joint,
            h_r_given_i_naive=h_r_given_i_naive,
            regime=regime,
            recommended_action=recommended_action,
            ambiguity_contribution=ambiguity_contribution,
            instability_contribution=instability_contribution,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "h_i": self.h_i,
            "h_r": self.h_r,
            "h_r_given_i": self.h_r_given_i,
            "h_joint": self.h_joint,
            "h_r_given_i_naive": self.h_r_given_i_naive,
            "regime": self.regime.value,
            "recommended_action": self.recommended_action,
            "ambiguity_contribution": self.ambiguity_contribution,
            "instability_contribution": self.instability_contribution,
        }


# Default thresholds instance for easy import
DEFAULT_THRESHOLDS = UncertaintyThresholds()
