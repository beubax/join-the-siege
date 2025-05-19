# Pytest fixtures and shared configuration for tests
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session as SQLAlchemySession # Renamed to avoid conflict
from fastapi.testclient import TestClient
import os
from pathlib import Path

from src.db.database import Base, get_db
from src.main import app # FastAPI app instance
from src.config import Settings # Adjusted import path

# Use a separate SQLite database for testing
TEST_DATABASE_URL = "sqlite:///:memory:" # In-memory SQLite for tests

engine = create_engine(
    TEST_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session", autouse=True)
def create_test_db_tables_session_scope():
    """Ensure all tables are created once per test session for in-memory DB."""
    from src.db import models # noqa Ensure models are registered
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine) # Optional: clean up at end of session if needed

@pytest.fixture(scope="function")
def db_session() -> SQLAlchemySession:
    """Yield a SQLAlchemy session for a test, with per-test transaction management."""
    # Create tables for each test function if they were dropped or if it's the first test
    # For in-memory, create_all() is cheap and ensures clean state.
    Base.metadata.create_all(bind=engine)
    
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()
    # Optionally, drop tables after each test if extreme isolation is needed, 
    # but create_all at start of each test with in-memory should be fine.
    # Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def override_get_db(db_session: SQLAlchemySession):
    def _override_get_db():
        try:
            yield db_session
        finally:
            pass # Session is managed by db_session fixture
    return _override_get_db

@pytest.fixture(scope="function")
def client(override_get_db):
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

@pytest.fixture
def test_settings() -> Settings:
    """Returns a fresh Settings instance for testing if needed."""
    return Settings()

@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "data"
    model_dir.mkdir(parents=True, exist_ok=True)
    synthetic_dir = model_dir / "synthetic_data"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    return model_dir 