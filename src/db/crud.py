from sqlalchemy.orm import Session
from typing import Optional, List
import uuid

from . import models
from .models import ClassificationJob, JobStatus, Feedback # Added Feedback model
from src.api import schemas # We'll define these Pydantic schemas soon

def create_classification_job(db: Session, job_id: str, filename: str, content_type: Optional[str] = None) -> models.ClassificationJob:
    db_job = models.ClassificationJob(
        id=job_id,
        original_filename=filename,
        file_content_type=content_type,
        status=JobStatus.PENDING
    )
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job

def get_classification_job(db: Session, job_id: str) -> Optional[models.ClassificationJob]:
    return db.query(models.ClassificationJob).filter(models.ClassificationJob.id == job_id).first()

def update_job_status(
    db: Session, 
    job_id: str, 
    status: JobStatus, 
    classification_result: Optional[str] = None, 
    error_message: Optional[str] = None,
    extracted_text: Optional[str] = None
) -> Optional[models.ClassificationJob]:
    db_job = get_classification_job(db, job_id)
    if db_job:
        db_job.status = status
        if classification_result is not None:
            db_job.classification_result = classification_result
        if error_message is not None:
            db_job.error_message = error_message
        if extracted_text is not None:
            db_job.extracted_text = extracted_text
        db.commit()
        db.refresh(db_job)
    return db_job

# --- Feedback CRUD ---

def create_feedback_entry(
    db: Session, 
    job: models.ClassificationJob, # Pass the job object
    corrected_classification: str,
    feedback_provider_id: Optional[str] = None,
    notes: Optional[str] = None
) -> models.Feedback:
    """Creates a new feedback entry in the database."""
    if not job.extracted_text:
        # This should ideally not happen if feedback is only allowed on jobs with text
        raise ValueError("Cannot record feedback for a job with no extracted text.")

    feedback_entry = models.Feedback(
        job_id=job.id,
        original_classification=job.classification_result, # The prediction before feedback
        corrected_classification=corrected_classification,
        extracted_text_at_feedback=job.extracted_text, # Text used for the original prediction
        feedback_provider_id=feedback_provider_id,
        notes=notes,
        is_processed_for_retraining=JobStatus.PENDING # Mark as pending for retraining
    )
    db.add(feedback_entry)
    db.commit()
    db.refresh(feedback_entry)
    return feedback_entry

def get_feedback_for_retraining(db: Session, limit: int = 100) -> List[models.Feedback]:
    """Retrieves feedback entries that are pending processing for retraining."""
    return db.query(models.Feedback).filter(
        models.Feedback.is_processed_for_retraining == JobStatus.PENDING
    ).limit(limit).all()

def mark_feedback_processed(
    db: Session, 
    feedback_ids: List[int], 
    status: JobStatus = JobStatus.COMPLETED # Mark as COMPLETED once used in retraining
) -> int:
    """Marks a list of feedback entries as processed for retraining."""
    if not feedback_ids:
        return 0
    num_updated = db.query(models.Feedback).filter(
        models.Feedback.id.in_(feedback_ids)
    ).update({"is_processed_for_retraining": status}, synchronize_session=False)
    db.commit()
    return num_updated

# Modified record_feedback for ClassificationJob to mainly update its own fields
# The creation of Feedback entry will be handled separately in the endpoint.
def update_job_after_feedback(
    db: Session, 
    job_id: str, 
    corrected_classification: str
) -> Optional[models.ClassificationJob]:
    """Updates the classification_result on the ClassificationJob itself after feedback."""
    db_job = get_classification_job(db, job_id)
    if db_job:
        db_job.classification_result = corrected_classification 
        # Optionally, change job status or add a flag like 'has_human_feedback' here if needed
        # For now, just updating the result to reflect the correction on the job itself.
        db.commit()
        db.refresh(db_job)
    return db_job

# def get_feedback_data_for_retraining(db: Session, limit: int = 100) -> List[models.ClassificationJob]:
#     """Retrieves jobs that have been corrected (implicitly, by having a classification_result 
#     that might have been updated by feedback) and could be used for retraining.
#     This is a simplified approach. A more robust system would explicitly mark feedback entries.
#     """
#     # This query is a placeholder. You'd need a more specific way to identify feedback.
#     # For example, if you added a `corrected_label` field or used the Feedback table.
#     return db.query(models.ClassificationJob).filter(
#         models.ClassificationJob.status == JobStatus.COMPLETED, # Assuming feedback is on completed jobs
#         models.ClassificationJob.classification_result.isnot(None)
#         # Add more conditions if you have a way to mark feedback, e.g., a boolean field `has_human_feedback`
#     ).limit(limit).all() 