from sqlalchemy import Column, Integer, String, DateTime, Enum as SQLAlchemyEnum, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from .database import Base # Assuming database.py will define Base

class JobStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class ClassificationJob(Base):
    __tablename__ = "classification_jobs"

    id = Column(String, primary_key=True, index=True) # Job ID from Celery task
    original_filename = Column(String, index=True)
    file_content_type = Column(String, nullable=True)
    status = Column(SQLAlchemyEnum(JobStatus), default=JobStatus.PENDING, index=True)
    extracted_text = Column(Text, nullable=True) # Added field for extracted text content
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    # Classification result
    # We can store a simple string or a more structured result (e.g. JSON with confidence scores)
    classification_result = Column(String, nullable=True) # Predicted category
    error_message = Column(Text, nullable=True)

    # Relationship to feedback (optional, for advanced feedback tracking)
    # feedback = relationship("Feedback", back_populates="job")

    def __repr__(self):
        return f"<ClassificationJob(id={self.id}, filename={self.original_filename}, status={self.status})>"

# Feedback table to store human corrections
class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, ForeignKey('classification_jobs.id'), nullable=False, index=True)
    original_classification = Column(String, nullable=True) # What the model first predicted for this job
    corrected_classification = Column(String, nullable=False, index=True)
    # Store the text that this feedback pertains to, for targeted retraining
    extracted_text_at_feedback = Column(Text, nullable=False)
    feedback_provider_id = Column(String, nullable=True) # Optional: if you track who provided feedback
    notes = Column(Text, nullable=True) # Optional: for any notes from the human reviewer
    provided_at = Column(DateTime(timezone=True), server_default=func.now())
    is_processed_for_retraining = Column(SQLAlchemyEnum(JobStatus, name="feedback_retraining_status"), default=JobStatus.PENDING) # PENDING, PROCESSING (during retraining), COMPLETED (used in a training run)

    # Relationship to the job this feedback is for
    job = relationship("ClassificationJob") # No back_populates needed if job doesn't directly link back to a list of feedbacks often

    def __repr__(self):
        return f"<Feedback(id={self.id}, job_id={self.job_id}, corrected={self.corrected_classification})>" 