import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import json

from src.classifier.synthetic_data import (
    _generate_detailed_prompt_for_industry,
    generate_and_store_synthetic_data_for_industry
)
from src.config import Settings # Adjusted import path

INDUSTRY_NAME = "test_industry"
NUM_DOCS = 2

@pytest.fixture
def mock_openai_client(mocker) -> MagicMock:
    mock_client_instance = MagicMock()
    mock_chat_completions = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    # Simulate the structure: client.chat.completions.create().choices[0].message.content
    mock_message.content = json.dumps({"documents": [
        {"document_text": f"Simulated text for doc 1 in {INDUSTRY_NAME}"},
        {"document_text": f"Simulated text for doc 2 in {INDUSTRY_NAME}"}
    ]})
    mock_choice.message = mock_message
    mock_chat_completions.create.return_value.choices = [mock_choice]
    mock_client_instance.chat.completions = mock_chat_completions
    # Patch the client instance in the synthetic_data module
    mocker.patch('src.classifier.synthetic_data.client', mock_client_instance)
    return mock_client_instance

def test_generate_detailed_prompt():
    prompt = _generate_detailed_prompt_for_industry(INDUSTRY_NAME, NUM_DOCS)
    assert INDUSTRY_NAME in prompt
    assert str(NUM_DOCS) in prompt
    assert "JSON list of strings" in prompt

def test_generate_and_store_data_success(
    mock_openai_client: MagicMock, temp_data_dir: Path, test_settings: Settings
):
    # Patch the SYNTHETIC_DATA_DIR in the synthetic_data module to use our temp dir
    # temp_data_dir fixture points to <tmp_path>/data/
    # The function will create <tmp_path>/data/synthetic_data/
    test_specific_synthetic_root = temp_data_dir / "synthetic_data"
    # Ensure the global settings.BASE_DIR is such that when synthetic_data.py calculates
    # its SYNTHETIC_DATA_DIR, it resolves to our test_specific_synthetic_root.
    # The easiest way is to patch `settings.BASE_DIR` for this test's scope if synthetic_data.py
    # re-evaluates `settings.BASE_DIR / "data" / "synthetic_data"` on each call.
    # Or, directly patch `SYNTHETIC_DATA_DIR` in the module.
    with patch('src.classifier.synthetic_data.SYNTHETIC_DATA_DIR', test_specific_synthetic_root):
        with patch('src.classifier.synthetic_data.settings', test_settings): # Ensure it uses test_settings
            test_settings.BASE_DIR = temp_data_dir.parent # so settings.BASE_DIR / data / synthetic_data works as expected

            stored_files = generate_and_store_synthetic_data_for_industry(INDUSTRY_NAME, NUM_DOCS)
            assert len(stored_files) == NUM_DOCS
            mock_openai_client.chat.completions.create.assert_called_once()

            expected_industry_path = test_specific_synthetic_root / INDUSTRY_NAME
            assert expected_industry_path.exists()
            assert len(list(expected_industry_path.glob("doc_*.txt"))) == NUM_DOCS
            for i, (file_path, label) in enumerate(stored_files):
                assert label == INDUSTRY_NAME
                assert file_path.name == f"doc_{i+1}.txt"
                assert file_path.exists()
                with open(file_path, "r") as f:
                    assert f"Simulated text for doc {i+1} in {INDUSTRY_NAME}" in f.read()

@patch('src.classifier.synthetic_data.client') # Mock the client directly
def test_generate_and_store_data_openai_error(mock_client, temp_data_dir: Path, test_settings: Settings):
    mock_client.chat.completions.create.side_effect = Exception("OpenAI API Error")
    test_specific_synthetic_root = temp_data_dir / "synthetic_data"
    with patch('src.classifier.synthetic_data.SYNTHETIC_DATA_DIR', test_specific_synthetic_root):
        with patch('src.classifier.synthetic_data.settings', test_settings):
            test_settings.BASE_DIR = temp_data_dir.parent
            stored_files = generate_and_store_synthetic_data_for_industry(INDUSTRY_NAME, NUM_DOCS)
            assert len(stored_files) == 0 