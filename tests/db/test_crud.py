import pytest
import uuid
from sqlalchemy.orm import Session as SQLAlchemySession # From conftest

from src.db import crud, models
from src.db.models import JobStatus

TEST_JOB_ID = str(uuid.uuid4())
TEST_FILENAME = "test_doc.txt"

def test_create_and_get_classification_job(db_session: SQLAlchemySession):
    job = crud.create_classification_job(db_session, job_id=TEST_JOB_ID, filename=TEST_FILENAME)
    assert job.id == TEST_JOB_ID
    assert job.original_filename == TEST_FILENAME
    assert job.status == JobStatus.PENDING

    retrieved_job = crud.get_classification_job(db_session, TEST_JOB_ID)
    assert retrieved_job is not None
    assert retrieved_job.id == TEST_JOB_ID

def test_update_job_status(db_session: SQLAlchemySession):
    job = crud.create_classification_job(db_session, job_id=str(uuid.uuid4()), filename="update_me.txt")
    updated_job = crud.update_job_status(
        db_session, 
        job_id=job.id, 
        status=JobStatus.COMPLETED, 
        classification_result="finance",
        extracted_text="Sample text."
    )
    assert updated_job is not None
    assert updated_job.status == JobStatus.COMPLETED
    assert updated_job.classification_result == "finance"
    assert updated_job.extracted_text == "Sample text."

def test_create_feedback_entry(db_session: SQLAlchemySession):
    job = crud.create_classification_job(db_session, job_id=str(uuid.uuid4()), filename="feedback_target.docx")
    # Simulate job has extracted text and an initial classification
    job.extracted_text = "This is the document text for feedback."
    job.classification_result = "initial_finance"
    db_session.add(job)
    db_session.commit()

    feedback = crud.create_feedback_entry(
        db_session, 
        job=job, 
        corrected_classification="corrected_legal"
    )
    assert feedback.job_id == job.id
    assert feedback.corrected_classification == "corrected_legal"
    assert feedback.original_classification == "initial_finance"
    assert feedback.extracted_text_at_feedback == job.extracted_text
    assert feedback.is_processed_for_retraining == JobStatus.PENDING

def test_get_feedback_for_retraining(db_session: SQLAlchemySession):
    job1 = crud.create_classification_job(db_session, job_id=str(uuid.uuid4()), filename="f1.txt", content_type="text/plain")
    job1.extracted_text = "text1"
    job1.classification_result = "cat1"
    db_session.add(job1)
    crud.create_feedback_entry(db_session, job=job1, corrected_classification="cat1_corrected")

    job2 = crud.create_classification_job(db_session, job_id=str(uuid.uuid4()), filename="f2.txt", content_type="text/plain")
    job2.extracted_text = "text2"
    job2.classification_result = "cat2"
    db_session.add(job2)
    feedback2 = crud.create_feedback_entry(db_session, job=job2, corrected_classification="cat2_corrected")
    # Manually mark one as processed to test filtering
    feedback2.is_processed_for_retraining = JobStatus.COMPLETED
    db_session.add(feedback2)
    db_session.commit()

    pending_feedback = crud.get_feedback_for_retraining(db_session, limit=10)
    assert len(pending_feedback) == 1
    assert pending_feedback[0].job_id == job1.id
    assert pending_feedback[0].is_processed_for_retraining == JobStatus.PENDING

def test_mark_feedback_processed(db_session: SQLAlchemySession):
    job = crud.create_classification_job(db_session, job_id=str(uuid.uuid4()), filename="f3.txt", content_type="text/plain")
    job.extracted_text = "text3"
    job.classification_result = "cat3"
    db_session.add(job)
    feedback = crud.create_feedback_entry(db_session, job=job, corrected_classification="cat3_corrected")
    db_session.commit()

    num_marked = crud.mark_feedback_processed(db_session, feedback_ids=[feedback.id], status=JobStatus.COMPLETED)
    assert num_marked == 1
    db_session.refresh(feedback)
    assert feedback.is_processed_for_retraining == JobStatus.COMPLETED

def test_update_job_after_feedback(db_session: SQLAlchemySession):
    job_id_for_update = str(uuid.uuid4())
    crud.create_classification_job(db_session, job_id=job_id_for_update, filename="update_job.txt")
    updated_job = crud.update_job_after_feedback(db_session, job_id=job_id_for_update, corrected_classification="new_corrected_label")
    assert updated_job is not None
    assert updated_job.classification_result == "new_corrected_label" 