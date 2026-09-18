from unittest.mock import patch

import pytest

from ascent_platform.external.medical_coder import MedicalCoder


@pytest.mark.asyncio
@patch('ascent_platform.external.medical_coder.AscentClient.query_ascent_api')
async def test_get_medical_codes_success(mock_query_ascent_api):
    # Arrange
    mock_response = {
        'query': [
            {
                'CONCEPT_DATA': {
                    'CONCEPT_ID': '123',
                    'CONCEPT_NAME': 'ConceptA',
                    'CONCEPT_CODE': 'C123',
                    'VOCABULARY_ID': 'VocabA',
                    'PATIENT_COUNT': 1000,
                },
                'SCORE': 0
            }
        ]
    }
    mock_query_ascent_api.return_value = mock_response
    medical_coder = MedicalCoder('client_id', 'client_secret')

    # Act
    result = await medical_coder.get_medical_codes('query')

    # Assert
    mock_query_ascent_api.assert_awaited_once()
    assert result == medical_coder.construct_response(mock_response)


@pytest.mark.asyncio
@patch('ascent_platform.external.medical_coder.AscentClient.query_ascent_api')
async def test_get_medical_codes_failure(mock_query_ascent_api):
    # Arrange
    mock_query_ascent_api.return_value = None
    medical_coder = MedicalCoder('client_id', 'client_secret')
    query = 'query'

    # Act
    result = await medical_coder.get_medical_codes(query)

    # Assert
    mock_query_ascent_api.assert_awaited_once()
    assert result == {query: []}


def test_construct_response():
    # Arrange
    medical_coder = MedicalCoder('client_id', 'client_secret')
    api_response = {
        'query': [
            {
                'CONCEPT_DATA':
                    {
                        'CONCEPT_ID': '123',
                        'CONCEPT_NAME': 'ConceptA',
                        'CONCEPT_CODE': 'C123',
                        'VOCABULARY_ID': 'VocabA',
                        'PATIENT_COUNT': 1000
                    },
                'SCORE': 0
            },
            {
                'CONCEPT_DATA':
                    {
                        'CONCEPT_ID': '456',
                        'CONCEPT_NAME': 'ConceptB',
                        'CONCEPT_CODE': 'C456',
                        'VOCABULARY_ID': 'VocabA',
                        'IS_VALID': False,
                        'PATIENT_COUNT': 50
                    },
                'SCORE': 0.5
            }
        ]
    }
    expected_response = {
        'query': [
            # IS_VALID defaults to True when the coder response predates the field
            {'CONCEPT_ID': '123', 'CONCEPT_NAME': 'ConceptA', 'CONCEPT_CODE': 'C123', 'VOCABULARY_ID': 'VocabA',
             'IS_VALID': True, 'SIMILARITY_SCORE': 0, 'PATIENT_COUNT': 1000},
            {'CONCEPT_ID': '456', 'CONCEPT_NAME': 'ConceptB', 'CONCEPT_CODE': 'C456', 'VOCABULARY_ID': 'VocabA',
             'IS_VALID': False, 'SIMILARITY_SCORE': 0.5, 'PATIENT_COUNT': 50}
        ]
    }

    # Act
    result = medical_coder.construct_response(api_response)

    # Assert
    assert result == expected_response


@pytest.mark.asyncio
@patch('ascent_platform.external.medical_coder.AscentClient.query_ascent_api')
async def test_get_patient_counts_omop_success(mock_query_ascent_api):
    """Test patient counts for an OMOP database returns expected results."""
    # Arrange
    mock_response = {
        "results": [
            {"concept_code": "E11.9", "concept_id": 201826, "vocabulary_id": "ICD10CM", "patient_count": 5000, "found": True},
            {"concept_code": "Z99.99", "concept_id": None, "vocabulary_id": "ICD10CM", "patient_count": None, "found": False},
        ],
        "database": "SYNTHETIC_EHR_OMOP",
        "codes_submitted": 2,
        "codes_found": 1,
    }
    mock_query_ascent_api.return_value = mock_response
    medical_coder = MedicalCoder('client_id', 'client_secret')

    codes = [{"code": "E11.9", "vocabulary_id": "ICD10CM"}, {"code": "Z99.99", "vocabulary_id": "ICD10CM"}]

    # Act
    result = await medical_coder.get_patient_counts(codes=codes, database="SYNTHETIC_EHR_OMOP")

    # Assert
    mock_query_ascent_api.assert_awaited_once_with(
        "patient-counts",
        payload={"codes": codes, "database": "SYNTHETIC_EHR_OMOP"},
    )
    assert result == mock_response
    assert result["codes_found"] == 1
    assert result["codes_submitted"] == 2


@pytest.mark.asyncio
@patch('ascent_platform.external.medical_coder.AscentClient.query_ascent_api')
async def test_get_patient_counts_with_schema(mock_query_ascent_api):
    """Test patient counts passes explicit schema to the API."""
    # Arrange
    mock_response = {
        "results": [
            {"concept_code": "E11.9", "concept_id": 201826, "vocabulary_id": "ICD10CM", "patient_count": 3000, "found": True},
        ],
        "database": "SYNTHETIC_EHR_OMOP",
        "codes_submitted": 1,
        "codes_found": 1,
    }
    mock_query_ascent_api.return_value = mock_response
    medical_coder = MedicalCoder('client_id', 'client_secret')

    codes = [{"code": "E11.9", "vocabulary_id": "ICD10CM"}]

    # Act
    result = await medical_coder.get_patient_counts(
        codes=codes, database="SYNTHETIC_EHR_OMOP", schema="CDM_2022Q4",
    )

    # Assert
    mock_query_ascent_api.assert_awaited_once_with(
        "patient-counts",
        payload={"codes": codes, "database": "SYNTHETIC_EHR_OMOP", "schema": "CDM_2022Q4"},
    )
    assert result == mock_response


@pytest.mark.asyncio
@patch('ascent_platform.external.medical_coder.AscentClient.query_ascent_api')
async def test_get_patient_counts_non_omop_success(mock_query_ascent_api):
    """Test patient counts for a non-OMOP database (concept_id is None)."""
    # Arrange
    mock_response = {
        "results": [
            {"concept_code": "E11.9", "concept_id": None, "vocabulary_id": "ICD10CM", "patient_count": 120, "found": True},
        ],
        "database": "MY_CUSTOM_DB",
        "codes_submitted": 1,
        "codes_found": 1,
    }
    mock_query_ascent_api.return_value = mock_response
    medical_coder = MedicalCoder('client_id', 'client_secret')

    codes = [{"code": "E11.9", "vocabulary_id": "ICD10CM"}]

    # Act
    result = await medical_coder.get_patient_counts(codes=codes, database="MY_CUSTOM_DB", schema="PUBLIC")

    # Assert
    mock_query_ascent_api.assert_awaited_once_with(
        "patient-counts",
        payload={"codes": codes, "database": "MY_CUSTOM_DB", "schema": "PUBLIC"},
    )
    assert result["codes_found"] == 1
    # Non-OMOP databases return None for concept_id
    assert result["results"][0]["concept_id"] is None


@pytest.mark.asyncio
@patch('ascent_platform.external.medical_coder.AscentClient.query_ascent_api')
async def test_get_patient_counts_with_standard_concept(mock_query_ascent_api):
    """Test patient counts with standard_concept flag passes it to the API."""
    mock_response = {
        "results": [],
        "database": "SYNTHETIC_EHR_OMOP",
        "codes_submitted": 1,
        "codes_found": 0,
    }
    mock_query_ascent_api.return_value = mock_response
    medical_coder = MedicalCoder('client_id', 'client_secret')

    codes = [{"code": "44054006", "vocabulary_id": "SNOMED"}]

    # Act
    result = await medical_coder.get_patient_counts(
        codes=codes, database="SYNTHETIC_EHR_OMOP", standard_concept="S",
    )

    # Assert
    mock_query_ascent_api.assert_awaited_once_with(
        "patient-counts",
        payload={
            "codes": codes,
            "database": "SYNTHETIC_EHR_OMOP",
            "standard_concept": "S",
        },
    )
    assert result == mock_response


@pytest.mark.asyncio
@patch('ascent_platform.external.medical_coder.AscentClient.query_ascent_api')
async def test_get_patient_counts_api_returns_none(mock_query_ascent_api):
    """Test patient counts graceful fallback when the API returns None."""
    # Arrange
    mock_query_ascent_api.return_value = None
    medical_coder = MedicalCoder('client_id', 'client_secret')

    codes = [{"code": "E11.9", "vocabulary_id": "ICD10CM"}]

    # Act
    result = await medical_coder.get_patient_counts(codes=codes, database="SYNTHETIC_EHR_OMOP")

    # Assert
    assert result == {
        "results": [],
        "database": "SYNTHETIC_EHR_OMOP",
        "codes_submitted": 1,
        "codes_found": 0,
    }


@pytest.mark.asyncio
@patch('ascent_platform.external.medical_coder.AscentClient.query_ascent_api')
async def test_get_patient_counts_omits_none_optional_params(mock_query_ascent_api):
    """Test that schema and standard_concept are omitted from payload when None."""
    mock_query_ascent_api.return_value = {
        "results": [], "database": "DB", "codes_submitted": 1, "codes_found": 0,
    }
    medical_coder = MedicalCoder('client_id', 'client_secret')
    codes = [{"code": "X", "vocabulary_id": "Y"}]

    # Act
    await medical_coder.get_patient_counts(codes=codes, database="DB")

    # Assert — payload should only have codes and database
    mock_query_ascent_api.assert_awaited_once_with(
        "patient-counts",
        payload={"codes": codes, "database": "DB"},
    )

