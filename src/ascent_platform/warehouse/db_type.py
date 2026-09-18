"""Utilities for classifying warehouse database types."""

_OMOP_SUFFIX = "_OMOP"


def is_omop_database(database_name: str) -> bool:
    """Return ``True`` if *database_name* refers to an OMOP-CDM database.

    The convention used across the platform is that OMOP databases have names
    that end with the ``_OMOP`` suffix (case-insensitive), e.g.
    ``SYNTHETIC_EHR_OMOP`` or ``SYNTHETIC_EHR_OMOP``.

    Args:
        database_name: The database name to test.

    Returns:
        ``True`` when the name ends with ``_OMOP`` (case-insensitive),
        ``False`` otherwise.
    """
    return database_name.upper().endswith(_OMOP_SUFFIX)
