import pytest
from pathlib import Path
import os

from src.config import Settings # Adjusted import path

# Note: The global `settings` object from src.config is created at import time.
# Testing environment variable overrides requires re-instantiating Settings or reloading the module.

def test_settings_default_values(test_settings: Settings):
    """Test that default settings are loaded correctly using a fresh instance."""
    s = test_settings # Uses the fixture which provides a fresh Settings instance
    assert s.PROJECT_NAME == "Heron File Classifier"
    assert "classifier.db" in s.DATABASE_URL
    assert s.ML_MODEL_TYPE in ["SGDClassifier", "MultinomialNB"] # Check it's one of the valid ones
    assert s.PREDICTION_CONFIDENCE_THRESHOLD is not None
    assert len(s.INITIAL_CATEGORIES) >= 5 # Check if it has the base categories + other
    assert "other" in s.INITIAL_CATEGORIES


def test_settings_model_dir_created_by_global_instance():
    """Test that the globally imported settings instance creates MODEL_DIR."""
    from src.config import settings as global_app_settings
    assert global_app_settings.MODEL_DIR.exists()
    assert global_app_settings.MODEL_DIR.is_dir()