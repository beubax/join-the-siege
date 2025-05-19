import os
from pathlib import Path
from typing import Optional

class Settings:
    PROJECT_NAME: str = "Heron File Classifier"
    VERSION: str = "0.1.0"

    BASE_DIR: Path = Path(__file__).resolve().parents[1] # Root of the project (join-the-siege/)
    
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'classifier.db'}")
    
    # ML Model settings
    MODEL_DIR: Path = BASE_DIR / "data"
    MODEL_NAME: str = "classifier_model.joblib"
    VECTORIZER_NAME: str = "vectorizer.joblib"
    MODEL_PATH: Path = MODEL_DIR / MODEL_NAME
    VECTORIZER_PATH: Path = MODEL_DIR / VECTORIZER_NAME
    ML_MODEL_TYPE: str = os.getenv("ML_MODEL_TYPE", "SGDClassifier") # Options: "SGDClassifier", "MultinomialNB"
    PREDICTION_CONFIDENCE_THRESHOLD: Optional[float] = float(os.getenv("PREDICTION_CONFIDENCE_THRESHOLD", "0.5")) # Threshold for assigning to 'other' if max confidence is too low. Set to None or 0 to disable.

    # Industry Mapping
    MAX_POTENTIAL_INDUSTRIES: int = int(os.getenv("MAX_POTENTIAL_INDUSTRIES", "100"))
    INDUSTRY_MAPPING_NAME: str = "industry_mapping.json"
    INDUSTRY_MAPPING_PATH: Path = MODEL_DIR / INDUSTRY_MAPPING_NAME

    # Transformer Escalation Settings
    ENABLE_TRANSFORMER_ESCALATION: bool = os.getenv("ENABLE_TRANSFORMER_ESCALATION", "True").lower() == "true"
    TRANSFORMER_MODEL_NAME: str = os.getenv("TRANSFORMER_MODEL_NAME", "facebook/bart-large-mnli")
    TRANSFORMER_CONFIDENCE_THRESHOLD: float = float(os.getenv("TRANSFORMER_CONFIDENCE_THRESHOLD", "0.7")) # Min confidence from transformer to override "other"

    # Celery settings
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

    # Supported file types and their parsers
    SUPPORTED_FILE_TYPES = {
        "application/pdf": "parse_pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "parse_docx", # .docx
        "text/plain": "parse_txt",
    }
    MAX_FILE_SIZE_MB: int = 25 # Maximum file size in MB

    # Initial categories for the classifier
    # These would ideally come from a more dynamic source or be part of model metadata
    INITIAL_CATEGORIES: list[str] = ["healthcare", "finance", "legal", "education", "technology", "other"]


settings = Settings()

# Ensure data directory exists
settings.MODEL_DIR.mkdir(parents=True, exist_ok=True) 