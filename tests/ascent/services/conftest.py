from unittest.mock import AsyncMock, MagicMock

import pytest

from ascent_domain.models.data_definitions import User


@pytest.fixture
def mock_rwd_db():
    db = MagicMock()
    db.execute = MagicMock()
    return db


@pytest.fixture
def mock_app_db():
    mock_result = MagicMock()
    mock_result.one = MagicMock()
    mock_result.all = MagicMock()
    db = AsyncMock()
    db.execute.return_value = mock_result
    db.scalars.return_value = mock_result
    db.commit = AsyncMock()
    db.add = MagicMock()
    db.delete = AsyncMock()
    return db


@pytest.fixture
def mock_user():
    return User(email="test@example.com")
