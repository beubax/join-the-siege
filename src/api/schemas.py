from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid

from src.db.models import JobStatus # Enum from our SQLAlchemy models

# Base model for common fields if any (optional here)
class JobBase(BaseModel):
    original_filename: Optional[str] = None
    file_content_type: Optional[str] = None

# Schema for response when a job is created
class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str = "File upload received. Processing started."
    original_filename: Optional[str] = None

# Schema for displaying job status and result
class JobStatusResponse(JobBase):
    id: str # Job ID
    status: JobStatus
    classification_result: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True # Changed from from_attributes = True for Pydantic v1 compatibility if needed, else use from_attributes

# Schema for submitting feedback
class FeedbackSchema(BaseModel):
    job_id: str = Field(..., description="The ID of the job for which feedback is being provided.")
    corrected_classification: str = Field(..., description="The human-verified correct classification label.")

# Schema for response after submitting feedback
class FeedbackResponse(BaseModel):
    message: str
    job_id: str
    corrected_classification: str

# Schema for requesting model retraining (optional, if we have an endpoint)
class RetrainRequest(BaseModel):
    force_retrain: bool = Field(False, description="Set to true to bypass any internal checks and force retraining.")
    retrain_on_feedback_only: bool = Field(False, description="If true, retrain only on data from the dedicated Feedback table. If false, retrain on all applicable classified jobs (which might include corrections). Default is false.")

# Schema for response of model retraining (optional)
class RetrainResponse(BaseModel):
    status: str
    message: str
    model_version: Optional[str] = None # If we version our models

# Schema for adding a new industry
class AddIndustryRequest(BaseModel):
    industry_name: str = Field(..., description="The name of the new industry to add (e.g., 'pharmaceutical').")
    # Optional: could allow specifying categories specific to this industry if needed later
    # categories: Optional[List[str]] = None 
    num_synthetic_documents: int = Field(10, description="Number of synthetic documents to generate for this new industry.", ge=1, le=50)

class AddIndustryResponse(BaseModel):
    message: str
    industry_name: str
    synthetic_data_generation_status: str # e.g., "Queued", "Completed", "Failed"
    retraining_status: str # e.g., "Queued", "Not Triggered", "Completed"
    # Optionally return task IDs for async operations
    data_generation_task_id: Optional[str] = None 
    retraining_task_id: Optional[str] = None 