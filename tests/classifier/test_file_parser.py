import pytest
from unittest.mock import patch, MagicMock

from src.classifier.file_parser import (
    parse_txt, extract_text_from_file,
    get_file_mime_type # parse_pdf, parse_docx are more complex to unit test simply without real files or heavy mocks
)
from src.config import Settings # Adjusted import path

DUMMY_TXT_CONTENT = "Simple text content."
DUMMY_TXT_BYTES = DUMMY_TXT_CONTENT.encode('utf-8')

def test_parse_txt():
    assert parse_txt(DUMMY_TXT_BYTES) == DUMMY_TXT_CONTENT

@patch('src.classifier.file_parser.magic.from_buffer')
def test_get_file_mime_type_success(mock_magic_from_buffer):
    mock_magic_from_buffer.return_value = "text/plain"
    assert get_file_mime_type(DUMMY_TXT_BYTES) == "text/plain"

@patch('src.classifier.file_parser.magic.from_buffer', side_effect=Exception("magic error"))
def test_get_file_mime_type_failure(mock_magic_from_buffer):
    assert get_file_mime_type(DUMMY_TXT_BYTES) is None

@patch('src.classifier.file_parser.get_file_mime_type')
@patch('src.classifier.file_parser.FILE_PARSERS')
def test_extract_text_from_file_txt(mock_file_parsers, mock_get_mime, test_settings: Settings):
    mock_get_mime.return_value = "text/plain"
    mock_txt_parser = MagicMock(return_value="Parsed TXT")
    # Ensure the settings point to a key that our mock_file_parsers will return
    test_settings.SUPPORTED_FILE_TYPES = {"text/plain": "parse_txt"}
    mock_file_parsers.get.return_value = mock_txt_parser # Mocking the .get() call directly
    
    result = extract_text_from_file(DUMMY_TXT_BYTES, "file.txt")
    assert result == "Parsed TXT"
    mock_file_parsers.get.assert_called_with("parse_txt")
    mock_txt_parser.assert_called_once_with(DUMMY_TXT_BYTES)

@patch('src.classifier.file_parser.get_file_mime_type')
def test_extract_text_from_file_unsupported(mock_get_mime, test_settings: Settings):
    mock_get_mime.return_value = "application/unsupported"
    test_settings.SUPPORTED_FILE_TYPES = {"text/plain": "parse_txt"} # Ensure unsupported type is not in settings
    with pytest.raises(ValueError, match="Unsupported or unrecognized file type"):
        extract_text_from_file(b"some bytes", "file.unsupported")

@patch('src.classifier.file_parser.get_file_mime_type', return_value=None) # MIME detection fails
@patch('src.classifier.file_parser.FILE_PARSERS')
def test_extract_text_from_file_fallback_to_extension(mock_file_parsers, mock_get_mime, test_settings: Settings):
    mock_txt_parser = MagicMock(return_value="Parsed TXT by extension")
    test_settings.SUPPORTED_FILE_TYPES = {"text/plain": "parse_txt"}
    mock_file_parsers.get.return_value = mock_txt_parser

    result = extract_text_from_file(DUMMY_TXT_BYTES, "document.txt") # .txt extension
    assert result == "Parsed TXT by extension"
    mock_file_parsers.get.assert_called_with("parse_txt")
    mock_txt_parser.assert_called_with(DUMMY_TXT_BYTES) 