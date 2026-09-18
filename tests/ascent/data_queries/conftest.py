from unittest.mock import AsyncMock, MagicMock

import pytest

from ascent_domain.models.data_definitions import User


@pytest.fixture
def mock_db():
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
def mock_current_user():
    return User(email="test@example.com")


@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    return session


@pytest.fixture
def mock_async_session_local(mock_session):
    mock = AsyncMock()
    mock.return_value.__aenter__.return_value = mock_session
    mock.return_value.__aexit__.return_value = None
    return mock
