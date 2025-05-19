import pytest
from unittest.mock import patch, MagicMock, call
import uuid
from pathlib import Path

# Adjusted import path for tasks
from src.celery_tasks.tasks import (
    process_file_classification,
    retrain_model_with_feedback_task,
    add_industry_and_retrain_task
)
from src.db.models import JobStatus, ClassificationJob, Feedback
from src.config import settings as app_settings # Actual app settings

@pytest.fixture
def mock_db_session_tasks(): # Renamed to avoid conflict if conftest is also imported here
    session = MagicMock()
    # Default return for query().filter().first()
    session.query.return_value.filter.return_value.first.return_value = None 
    # Default return for query().filter().limit().all()
    session.query.return_value.filter.return_value.limit.return_value.all.return_value = []
    return session

@pytest.fixture
def mock_celery_task_context():
    task = MagicMock()
    task.request.id = "test_celery_task_id"
    return task

@patch('src.celery_tasks.tasks.SessionLocal')
@patch('src.celery_tasks.tasks.crud')
@patch('src.celery_tasks.tasks.extract_text_from_file')
@patch('src.celery_tasks.tasks.classifier_instance')
@patch('src.celery_tasks.tasks.open', new_callable=MagicMock)
@patch('src.celery_tasks.tasks.os.path.exists', MagicMock(return_value=True))
@patch('src.celery_tasks.tasks.os.remove', MagicMock())
def test_process_file_classification_simple_success(
    mock_open_file, mock_classifier, mock_extract_text, mock_crud, mock_session_local,
    mock_db_session_tasks, mock_celery_task_context
):
    mock_session_local.return_value = mock_db_session_tasks
    mock_extract_text.return_value = "Test text"
    mock_classifier.is_fitted.return_value = True
    mock_classifier.predict.return_value = "finance"
    mock_open_file.return_value.__enter__.return_value.read.return_value = b"file data"

    job_id = str(uuid.uuid4())
    process_file_classification(mock_celery_task_context, job_id, "/fake/path.txt", "file.txt")

    mock_extract_text.assert_called_once_with(b"file data", original_filename="/fake/path.txt")
    mock_classifier.predict.assert_called_once_with("Test text")
    # Check the final status update
    update_calls = mock_crud.update_job_status.call_args_list
    final_call_args = update_calls[-1][0]
    final_call_kwargs = update_calls[-1][1]
    assert final_call_kwargs['status'] == JobStatus.COMPLETED
    assert final_call_kwargs['classification_result'] == "finance"
    assert final_call_kwargs['extracted_text'] == "Test text"
    mock_db_session_tasks.close.assert_called_once()

@patch('src.celery_tasks.tasks.SessionLocal')
@patch('src.celery_tasks.tasks.crud')
@patch('src.celery_tasks.tasks.classifier_instance')
def test_retrain_model_with_feedback_task_feedback_only(
    mock_classifier, mock_crud, mock_session_local, mock_db_session_tasks
):
    mock_session_local.return_value = mock_db_session_tasks
    mock_feedback_entry = Feedback(extracted_text_at_feedback="feedback text", corrected_classification="legal", id=1)
    mock_crud.get_feedback_for_retraining.return_value = [mock_feedback_entry]

    retrain_model_with_feedback_task(retrain_on_feedback_only=True)

    mock_crud.get_feedback_for_retraining.assert_called_once()
    mock_classifier.train.assert_called_once_with(["feedback text"], ["legal"], is_initial_training=False)
    mock_crud.mark_feedback_processed.assert_called_once_with(mock_db_session_tasks, feedback_ids=[1], status=JobStatus.COMPLETED)
    mock_db_session_tasks.close.assert_called_once()

@patch('src.celery_tasks.tasks.SessionLocal')
@patch('src.celery_tasks.tasks.classifier_instance')
def test_retrain_model_with_feedback_task_all_data(
    mock_classifier, mock_session_local, mock_db_session_tasks
):
    mock_session_local.return_value = mock_db_session_tasks
    mock_job = ClassificationJob(extracted_text="job text", classification_result="finance")
    # Mock the direct query made by the task when not feedback_only
    mock_db_session_tasks.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [mock_job]

    retrain_model_with_feedback_task(retrain_on_feedback_only=False)
    
    mock_classifier.train.assert_called_once_with(["job text"], ["finance"], is_initial_training=False)
    mock_db_session_tasks.close.assert_called_once()