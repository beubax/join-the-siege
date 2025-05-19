from fastapi import APIRouter, Depends, FastAPI, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
import shutil # For saving uploaded file temporarily
import tempfile
import uuid
import os
from pathlib import Path
from contextlib import asynccontextmanager
from src.db import crud, models
from src.db.database import get_db, create_db_tables # For initial table creation
from src.api import schemas
from src.celery_tasks.tasks import process_file_classification, retrain_model_with_feedback_task, add_industry_and_retrain_task # Added new task
from src.config import settings
from src.classifier.file_parser import get_file_mime_type # For quick validation
from src.classifier.ml_model import classifier_instance, train_initial_model_from_scratch # For initial training check


# Ensure the directory for temporary file uploads exists
TEMP_UPLOAD_DIR = settings.BASE_DIR / "temp_uploads"
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create database tables if they don't exist
    # In a production environment, you'd use Alembic migrations.
    create_db_tables()
    print("Database tables checked/created.")
    
    # Check if model is trained, if not, train a basic one
    if not classifier_instance.is_fitted():
        print("Classifier model not found or not fitted. Training initial model...")
        # This uses the dummy data logic in ml_model.py
        train_initial_model_from_scratch()
        # Reload the instance to ensure it picks up the trained model
        classifier_instance._load_model()
        classifier_instance._load_vectorizer()
        if classifier_instance.is_fitted():
            print("Initial model training complete.")
        else:
            print("Error: Initial model training failed on startup.")
            # Consider raising an error or specific handling if startup training fails
    yield
    # Clean up resources

router = APIRouter(lifespan=lifespan)

@router.post("/classify_file/", response_model=schemas.JobCreateResponse)
async def classify_file_endpoint(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Accepts a file upload, saves it temporarily, creates a classification job,
    and queues it for processing via Celery.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    # Validate file size
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    await file.seek(0) # Reset file pointer
    if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB."
        )

    # Validate file type (basic check using python-magic on first few bytes if needed, or rely on parser)
    # For a more robust check without reading the whole file into memory immediately for mime:
    # temp_peek_bytes = await file.read(2048) # Read first 2KB for MIME check
    # await file.seek(0) # Reset pointer
    # detected_mime = get_file_mime_type(temp_peek_bytes)
    # if detected_mime not in settings.SUPPORTED_FILE_TYPES:
    #     raise HTTPException(status_code=415, detail=f"Unsupported file type: {detected_mime}. Supported: {list(settings.SUPPORTED_FILE_TYPES.keys())}")
    # For now, we let the Celery task handle detailed parsing and type errors.
    
    job_id = str(uuid.uuid4())
    
    # Save the uploaded file to a temporary location for the Celery task to access
    # The Celery task will be responsible for deleting this file after processing.
    temp_file_path = TEMP_UPLOAD_DIR / f"{job_id}_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save uploaded file: {e}")
    finally:
        await file.close()
    # Check if the file was successfully saved
    if not os.path.exists(temp_file_path):
        raise HTTPException(status_code=500, detail="Failed to save the uploaded file.")
    else:
        print(f"File saved to {temp_file_path}")
    # Create job entry in DB
    db_job = crud.create_classification_job(
        db, 
        job_id=job_id, 
        filename=file.filename,
        content_type=file.content_type # Use MIME type from upload
    )

    # Send task to Celery worker
    process_file_classification.delay(
        job_id=db_job.id, 
        file_path=str(temp_file_path), 
        original_filename=db_job.original_filename, 
        content_type=db_job.file_content_type
    )
    
    return schemas.JobCreateResponse(
        job_id=db_job.id, 
        status=db_job.status, 
        original_filename=db_job.original_filename
    )

@router.get("/jobs/{job_id}", response_model=schemas.JobStatusResponse)
def get_job_status_endpoint(job_id: str, db: Session = Depends(get_db)):
    """Retrieves the status and result of a classification job."""
    db_job = crud.get_classification_job(db, job_id=job_id)
    if db_job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return db_job

@router.post("/feedback/", response_model=schemas.FeedbackResponse)
def submit_feedback_endpoint(
    feedback_data: schemas.FeedbackSchema,
    db: Session = Depends(get_db)
):
    """
    Allows submission of human-verified feedback for a classified document.
    This simplified version updates the job record directly.
    A more advanced version might store feedback separately and trigger retraining differently.
    """
    db_job = crud.get_classification_job(db, job_id=feedback_data.job_id)
    if db_job is None:
        raise HTTPException(status_code=404, detail=f"Job with ID {feedback_data.job_id} not found.")

    if db_job.status != models.JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Feedback can only be provided for completed jobs. Current status: {db_job.status}")

    if not db_job.extracted_text:
        raise HTTPException(status_code=400, detail=f"Cannot provide feedback for job {feedback_data.job_id} as it has no extracted text stored.")

    # 1. Create a new entry in the Feedback table
    try:
        feedback_entry = crud.create_feedback_entry(
            db,
            job=db_job, # Pass the full job object
            corrected_classification=feedback_data.corrected_classification
            # Can add provider_id or notes from feedback_data if schema supports it
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Log this error
        raise HTTPException(status_code=500, detail=f"Failed to create feedback entry: {e}")

    # 2. Update the ClassificationJob's result to reflect the feedback (as per existing behavior)
    updated_job = crud.update_job_after_feedback(db, job_id=feedback_data.job_id, corrected_classification=feedback_data.corrected_classification)
    
    if not updated_job or not feedback_entry: # Should not happen if job was found and feedback created
        raise HTTPException(status_code=500, detail="Failed to fully record feedback.")

    return schemas.FeedbackResponse(
        message="Feedback received, recorded in feedback log, and original job updated.",
        job_id=updated_job.id,
        corrected_classification=updated_job.classification_result 
    )

@router.post("/industries/", response_model=schemas.AddIndustryResponse, status_code=202) # 202 Accepted
async def add_new_industry_endpoint(
    payload: schemas.AddIndustryRequest,
    # background_tasks: BackgroundTasks # For truly async, Celery is better
    db: Session = Depends(get_db) # db might be used if we create a job for this task
):
    """
    Adds a new industry: triggers synthetic data generation and model retraining as a background task.
    """
    # Optional: Check if industry already exists in settings.INITIAL_CATEGORIES or a dynamic list
    # if payload.industry_name in settings.INITIAL_CATEGORIES:
    #     # Or just proceed, as retraining will just add more data for that class
    #     pass 
        # raise HTTPException(status_code=409, detail=f"Industry '{payload.industry_name}' already configured or being processed.")

    # For simplicity, we don't create a DB job for this task, but one could be added for tracking.
    
    task = add_industry_and_retrain_task.delay(payload.industry_name, payload.num_synthetic_documents)
    
    # It might be good to add the new industry to a dynamic list/DB table of known industries 
    # if settings.INITIAL_CATEGORIES isn't meant to be the sole source of truth long-term.
    # For now, the model itself will learn the new class.
    # If settings.INITIAL_CATEGORIES needs to be updated to reflect all known classes for other parts
    # of the system (e.g. UI dropdowns), this would require a different mechanism than just appending at runtime.
    # One approach: store known labels in the DB or a config file that can be updated.
    if payload.industry_name not in settings.INITIAL_CATEGORIES:
        # This runtime append is for the current session, won't persist across restarts unless settings are saved.
        # However, the model training in ml_model.py handles new classes dynamically anyway.
        # settings.INITIAL_CATEGORIES.append(payload.industry_name)
        print(f"New industry '{payload.industry_name}' will be processed. It may be added to the model's known classes.")

    return schemas.AddIndustryResponse(
        message=f"Request to add industry '{payload.industry_name}' received. Processing in background.",
        industry_name=payload.industry_name,
        synthetic_data_generation_status="Queued",
        retraining_status="Queued with data generation",
        # data_generation_task_id=task.id, # The task ID is for the combined task
        # retraining_task_id=task.id 
    )

@router.post("/retrain_model/", response_model=schemas.RetrainResponse)
async def trigger_retraining_endpoint(payload: schemas.RetrainRequest):
    """
    Manually triggers the model retraining task.
    Allows choosing between retraining on dedicated feedback data or all classified data.
    """
    task = retrain_model_with_feedback_task.delay(
        retrain_on_feedback_only=payload.retrain_on_feedback_only
    )
    return schemas.RetrainResponse(
        status="Retraining Initiated", 
        message=f"Model retraining task has been queued. Mode: {'Feedback-Only' if payload.retrain_on_feedback_only else 'All Corrected/Classified Data'}. Check Celery worker logs for progress.",
        # task_id=task.id 
    ) 