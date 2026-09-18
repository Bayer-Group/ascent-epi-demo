"""Tests for ascent_platform.snowflake.db_type utilities."""


from ascent_platform.warehouse.db_type import is_omop_database


class TestIsOmopDatabase:
    """Tests for the is_omop_database helper."""

    def test_returns_true_for_standard_omop_database(self):
        assert is_omop_database("SYNTHETIC_EHR_OMOP") is True

    def test_returns_true_for_omop_database_lowercase(self):
        assert is_omop_database("synthetic_claims_omop") is True

    def test_returns_true_for_omop_database_mixed_case(self):
        assert is_omop_database("Synthetic_Claims_Omop") is True

    def test_returns_true_for_ehr_omop_database(self):
        assert is_omop_database("SYNTHETIC_EHR_OMOP") is True

    def test_returns_true_for_another_omop_database(self):
        assert is_omop_database("SYNTHETIC_EHR_OMOP") is True

    def test_returns_false_for_non_omop_database(self):
        assert is_omop_database("CLAIMS_DATABASE") is False

    def test_returns_false_for_database_with_omop_not_at_end(self):
        assert is_omop_database("OMOP_DATA_WAREHOUSE") is False

    def test_returns_false_for_empty_string(self):
        assert is_omop_database("") is False

    def test_returns_false_for_database_with_omop_in_middle(self):
        assert is_omop_database("MY_OMOP_DATABASE_V2") is False
