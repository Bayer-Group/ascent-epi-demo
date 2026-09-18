"""The OMOP pipeline's view of the medical coder.

This subclass exists for a single difference from
``ascent_platform.external.medical_coder``: the platform client adds
``IS_VALID`` to every concept and this one does not.

Keep it that way unless someone decides otherwise on purpose. These concepts
feed ``database_concepts`` and from there placeholder substitution on the OMOP
QA path, so an extra key is not provably inert -- it can reach a prompt.
"""

import logging
from typing import Any

from ascent_platform.external.medical_coder import MedicalCoder as _PlatformMedicalCoder
from ascent_platform.external.medical_coder import unpack_concept

logger = logging.getLogger(__name__)


class MedicalCoder(_PlatformMedicalCoder):
    def extract_concept_data(self, concept_data: list) -> list[dict[str, Any]]:
        """As the platform's, minus IS_VALID. See the module docstring."""
        return [unpack_concept(concept) for concept in concept_data]
