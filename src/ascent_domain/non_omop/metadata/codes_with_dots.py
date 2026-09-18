from collections import Counter

from ascent_domain.non_omop.schemas.sql_creation_simplified import SQLCreationSimplified


def do_medical_entities_contain_dots(sql_preparation: SQLCreationSimplified):
    """
    Determines the most common value of 'codes_with_dots' among entity references in a SQLCreationSimplified object.

    Args:
        sql_preparation (SQLCreationSimplified): An object containing extracted medical concepts and entity references.

    Returns:
        Any: The most common value of 'codes_with_dots' among the entity references, or None if there are no values.

    Notes:
        - Returns None if 'entity_references' is empty or if no 'codes_with_dots' values are present.
        - The type of the returned value depends on the type of 'codes_with_dots' (commonly bool, str, or int).
    """
    medical_coding = sql_preparation.extracted_concepts.entity_references
    codes_with_dots_values = [
        entry.codes_with_dots
        for entry in medical_coding
    ]

    if not codes_with_dots_values:
        return None

    frequency = Counter(codes_with_dots_values)
    # most_common returns list like [(value, count)]
    return frequency.most_common(1)[0][0]
