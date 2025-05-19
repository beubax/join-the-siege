import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session as SQLAlchemySession # from conftest
from unittest.mock import patch, MagicMock, ANY
import uuid
import os
from pathlib import Path

from src.db import crud, models
from src.db.models import JobStatus
from src.config import settings as app_settings # Actual app settings

def test_health_check_api(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert "Heron File Classifier is running" in response.json()["message"]

@patch('src.api.endpoints.process_file_classification.delay')
def test_classify_file_endpoint(mock_celery_delay, client: TestClient, db_session: SQLAlchemySession, temp_data_dir: Path):
    # For this test, let's ensure the TEMP_UPLOAD_DIR in endpoints points to a test-managed temporary directory
    test_temp_uploads = temp_data_dir.parent / "api_temp_uploads"
    test_temp_uploads.mkdir(parents=True, exist_ok=True)

    with patch('src.api.endpoints.TEMP_UPLOAD_DIR', test_temp_uploads):
        file_content = b"dummy pdf content"
        files = {"file": ("test_upload.pdf", file_content, "application/pdf")}
        response = client.post("/api/classify_file/", files=files)

        assert response.status_code == 200
        json_response = response.json()
        job_id = json_response["job_id"]
        assert json_response["status"] == JobStatus.PENDING.value

        # Verify DB record
        db_job = crud.get_classification_job(db_session, job_id)
        assert db_job is not None
        assert db_job.original_filename == "test_upload.pdf"

        # Verify Celery task call and temp file creation
        expected_temp_file_path = test_temp_uploads / f"{job_id}_test_upload.pdf"
        mock_celery_delay.assert_called_once_with(
            job_id=job_id,
            file_path=str(expected_temp_file_path),
            original_filename="test_upload.pdf",
            content_type="application/pdf"
        )
        assert expected_temp_file_path.exists()
        # Clean up the temp file as the Celery task is mocked
        if expected_temp_file_path.exists():
            os.remove(expected_temp_file_path)

def test_get_job_status_api(client: TestClient, db_session: SQLAlchemySession):
    job_id = str(uuid.uuid4())
    crud.create_classification_job(db_session, job_id=job_id, filename="status_test.txt")
    crud.update_job_status(db_session, job_id=job_id, status=JobStatus.COMPLETED, classification_result="legal")
    
    response = client.get(f"/api/jobs/{job_id}")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["id"] == job_id
    assert json_response["status"] == JobStatus.COMPLETED.value
    assert json_response["classification_result"] == "legal"

def test_get_job_status_not_found_api(client: TestClient):
    response = client.get(f"/api/jobs/{str(uuid.uuid4())}")
    assert response.status_code == 404

@patch('src.api.endpoints.add_industry_and_retrain_task.delay')
def test_add_new_industry_api(mock_celery_add_industry, client: TestClient):
    payload = {"industry_name": "gaming", "num_synthetic_documents": 7}
    response = client.post("/api/industries/", json=payload)
    assert response.status_code == 202
    assert response.json()["industry_name"] == "gaming"
    mock_celery_add_industry.assert_called_once_with("gaming", 7)

@patch('src.api.endpoints.retrain_model_with_feedback_task.delay')
def test_trigger_retraining_api(mock_celery_retrain, client: TestClient):
    payload = {"retrain_on_feedback_only": True}
    response = client.post("/api/retrain_model/", json=payload)
    assert response.status_code == 200
    assert "Feedback-Only" in response.json()["message"]
    mock_celery_retrain.assert_called_once_with(retrain_on_feedback_only=True)

def test_submit_feedback_api(client: TestClient, db_session: SQLAlchemySession):
    job_id = str(uuid.uuid4())
    # Create a job and mark it as completed with some text
    job = crud.create_classification_job(db_session, job_id=job_id, filename="feedback_doc.pdf")
    crud.update_job_status(db_session, job_id=job_id, status=JobStatus.COMPLETED, 
                            classification_result="initial_guess", extracted_text="Some text for feedback.")

    feedback_payload = {"job_id": job_id, "corrected_classification": "final_label"}
    response = client.post("/api/feedback/", json=feedback_payload)
    assert response.status_code == 200
    assert response.json()["corrected_classification"] == "final_label"

    # Check that feedback was created in DB
    feedback_entries = db_session.query(models.Feedback).filter(models.Feedback.job_id == job_id).all()
    assert len(feedback_entries) == 1
    assert feedback_entries[0].corrected_classification == "final_label"
    assert feedback_entries[0].extracted_text_at_feedback == "Some text for feedback."
    # Check that original job was updated
    updated_job = crud.get_classification_job(db_session, job_id)
    assert updated_job.classification_result == "final_label" 