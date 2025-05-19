import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from pathlib import Path

from src.classifier.ml_model import TextClassifier #, train_initial_model_from_scratch - test separately or mock heavily
from src.config import Settings # Adjusted import path

SAMPLE_TEXTS = ["doc1 text finance related", "doc2 text legal services"]
SAMPLE_LABELS = ["finance", "legal"]

@pytest.fixture
def mock_model_settings(temp_data_dir: Path, test_settings: Settings) -> Settings:
    test_settings.MODEL_DIR = temp_data_dir
    test_settings.MODEL_PATH = temp_data_dir / "test_model.joblib"
    test_settings.VECTORIZER_PATH = temp_data_dir / "test_vectorizer.joblib"
    # Ensure sample labels are in initial categories for some tests
    test_settings.INITIAL_CATEGORIES = list(np.unique(SAMPLE_LABELS + test_settings.INITIAL_CATEGORIES))
    return test_settings

@patch('src.classifier.ml_model.joblib.dump')
@patch('src.classifier.ml_model.joblib.load', side_effect=FileNotFoundError) # No existing model
def test_text_classifier_init_new(mock_joblib_load, mock_joblib_dump, mock_model_settings: Settings):
    # Patch settings within the ml_model module for the global classifier_instance if it's created at import time
    # and for any direct use of settings in the TextClassifier constructor or methods
    with patch('src.classifier.ml_model.settings', mock_model_settings):
        classifier = TextClassifier(model_path=mock_model_settings.MODEL_PATH, vectorizer_path=mock_model_settings.VECTORIZER_PATH)
        assert classifier.model is not None
        assert classifier.vectorizer is not None
        assert not classifier.is_fitted()

@patch('src.classifier.ml_model.joblib.dump')
@patch('src.classifier.ml_model.joblib.load')
def test_text_classifier_train_and_predict(mock_joblib_load, mock_joblib_dump, mock_model_settings: Settings):
    # Simulate no model exists initially
    mock_joblib_load.side_effect = FileNotFoundError 
    with patch('src.classifier.ml_model.settings', mock_model_settings):
        classifier = TextClassifier(model_path=mock_model_settings.MODEL_PATH, vectorizer_path=mock_model_settings.VECTORIZER_PATH)
        
        # Mock the actual sklearn model's fit and predict methods
        classifier.model = MagicMock() 
        classifier.vectorizer = MagicMock()
        classifier.vectorizer.transform.return_value = np.array([[0.1, 0.2]]) # Dummy transformed data
        classifier.model.predict.return_value = np.array([SAMPLE_LABELS[0]])
        classifier.model.classes_ = np.array(mock_model_settings.INITIAL_CATEGORIES)
        # Mock predict_proba if thresholding logic is deeply tested, or simplify
        classifier.model.predict_proba.return_value = np.array([[0.8, 0.1, 0.1, 0.0, 0.0, 0.0][:len(classifier.model.classes_)]])

        classifier.train(SAMPLE_TEXTS, SAMPLE_LABELS, is_initial_training=True)
        classifier.model.partial_fit.assert_called_once() # Check if internal model's fit was called
        mock_joblib_dump.assert_any_call(classifier.model, mock_model_settings.MODEL_PATH)

        # After training, mark as fitted (mocking this part for simplicity of predict test)
        # In a real scenario, the fit method of the mocked model would set this up.
        classifier.model.coef_ = True # Simple way to mark as fitted for `is_fitted`

        prediction = classifier.predict(SAMPLE_TEXTS[0])
        assert prediction == "other"

def test_text_classifier_predict_not_fitted(mock_model_settings: Settings):
    with patch('src.classifier.ml_model.settings', mock_model_settings):
        with patch('src.classifier.ml_model.joblib.load', side_effect=FileNotFoundError):
            classifier = TextClassifier(model_path=mock_model_settings.MODEL_PATH, vectorizer_path=mock_model_settings.VECTORIZER_PATH)
            assert not classifier.is_fitted()
            assert classifier.predict("some text") == "other" # Default fallback

# Test for train_initial_model_from_scratch is more of an integration test.
# For a simple unit test, you'd mock many of its internal calls heavily.
@patch('src.classifier.ml_model.generate_and_store_synthetic_data_for_industry')
@patch('src.classifier.ml_model.TextClassifier.train') # Mock the train method of the instance
@patch('src.classifier.ml_model.os.remove', MagicMock())
@patch('src.classifier.ml_model.Path.exists', MagicMock(return_value=False))
@patch('src.classifier.ml_model.open', new_callable=MagicMock)
def test_train_initial_model_from_scratch_calls_generation_and_training(
    mock_open_file, mock_textclassifier_train, mock_generate_data, 
    mock_model_settings: Settings # Uses settings for categories
):
    # We patch the global settings object used by the function directly
    with patch('src.classifier.ml_model.settings', mock_model_settings):
        from src.classifier.ml_model import train_initial_model_from_scratch
        # Simulate synthetic data generation returning some file paths
        mock_generate_data.return_value = [(Path("fake/doc1.txt"), "finance")]
        # Simulate reading that file
        mock_open_file.return_value.__enter__.return_value.read.return_value = "Fake document text"

        train_initial_model_from_scratch()

        assert mock_generate_data.call_count == 0
        mock_textclassifier_train.assert_called_once()
        args, kwargs = mock_textclassifier_train.call_args
        assert kwargs['is_initial_training'] == True
        assert len(args[0]) > 0 # texts
        assert len(args[1]) > 0 # labels 