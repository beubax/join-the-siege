from src.celery_tasks.celery_app import celery_app
from src.db.database import SessionLocal # To create a new session for the task
from src.db import crud
from src.db import models
from src.classifier.file_parser import extract_text_from_file
from src.classifier.ml_model import classifier_instance # The global classifier instance
from src.config import settings
import tempfile
import os
from pathlib import Path
from src.db.models import JobStatus, ClassificationJob, Feedback # Added Feedback

@celery_app.task(bind=True, name="tasks.process_file_classification")
def process_file_classification(self, job_id: str, file_path: str, original_filename: str, content_type: str):
    """ 
    Celery task to process file classification.
    Receives a file_path to a temporary file, its original name, and content type.
    """
    db = SessionLocal()
    try:
        print(f"[Task {self.request.id}] Starting processing for job_id: {job_id}, file: {original_filename}")
        crud.update_job_status(db, job_id=job_id, status=models.JobStatus.PROCESSING)
        
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        # 1. Extract text
        extracted_text = extract_text_from_file(file_bytes, original_filename=original_filename)
        
        # Store extracted text early if available, regardless of classification outcome (unless parsing failed hard)
        if extracted_text:
            crud.update_job_status(db, job_id=job_id, status=models.JobStatus.PROCESSING, extracted_text=extracted_text) # Update with text
        else: # If no text, status remains PROCESSING, but no text is stored yet
            crud.update_job_status(db, job_id=job_id, status=models.JobStatus.PROCESSING)

        classification = "other" # Default if issues occur
        if not extracted_text or extracted_text.strip() == "":
            print(f"[Task {self.request.id}] No text could be extracted from {original_filename}.")
            # Classification remains 'other', status will be COMPLETED with this result
            # Or you might want to set to FAILED if text extraction is mandatory for a valid classification
            # crud.update_job_status(db, job_id=job_id, status=JobStatus.FAILED, error_message="No text content extracted.")
        else:
             # 2. Classify text
            if not classifier_instance.is_fitted():
                print(f"[Task {self.request.id}] Model not fitted. Attempting to train initial model.")
                # This is a fallback, ideally the model should be trained beforehand.
                # For a production system, you might raise an alert or fail the task.
                from src.classifier.ml_model import train_initial_model_from_scratch
                train_initial_model_from_scratch() # Train with default dummy data
                # Reload classifier instance to ensure it picks up the newly trained model
                classifier_instance._load_model() # Private access for this specific recovery scenario
                classifier_instance._load_vectorizer()
                if not classifier_instance.is_fitted():
                    crud.update_job_status(db, job_id=job_id, status=models.JobStatus.FAILED, error_message="Model not fitted and initial training failed.", extracted_text=extracted_text if extracted_text else "")
                    return

            classification = classifier_instance.predict(extracted_text)
        
        # 3. Update database with final status and classification result
        crud.update_job_status(db, job_id=job_id, status=models.JobStatus.COMPLETED, classification_result=classification, extracted_text=extracted_text if extracted_text else "")
        print(f"[Task {self.request.id}] Successfully classified {original_filename} as {classification}. Job ID: {job_id}")

    except ValueError as ve: # Catching specific errors from file_parser
        print(f"[Task {self.request.id}] ValueError during processing job {job_id}: {ve}")
        # Try to get text if it was extracted before error for logging/DB storage
        current_text = locals().get('extracted_text')
        crud.update_job_status(db, job_id=job_id, status=models.JobStatus.FAILED, error_message=str(ve), extracted_text=current_text if current_text else "Parsing error before text extraction")
    except ImportError as ie: # Catching missing parsers
        print(f"[Task {self.request.id}] ImportError during processing job {job_id}: {ie}")
        crud.update_job_status(db, job_id=job_id, status=models.JobStatus.FAILED, error_message=str(ie))
    except Exception as e:
        print(f"[Task {self.request.id}] Error processing job {job_id}: {e}")
        # Potentially retry logic here for certain types of errors
        # self.retry(exc=e, countdown=60, max_retries=3)
        current_text = locals().get('extracted_text')
        crud.update_job_status(db, job_id=job_id, status=models.JobStatus.FAILED, error_message=str(e), extracted_text=current_text if current_text else "Error before/during text extraction")
    finally:
        db.close()
        # Clean up the temporary file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"[Task {self.request.id}] Cleaned up temporary file: {file_path}")
        except Exception as e_clean:
            print(f"[Task {self.request.id}] Error cleaning up temporary file {file_path}: {e_clean}")

@celery_app.task(name="tasks.retrain_model_with_feedback")
def retrain_model_with_feedback_task(retrain_on_feedback_only: bool = False):
    """Celery task to retrain the model. 
    If retrain_on_feedback_only is True, uses data from the Feedback table.
    Otherwise, uses all relevant data from ClassificationJob table (which includes corrections).
    """
    db = SessionLocal()
    texts_for_retraining = []
    labels_for_retraining = []
    processed_feedback_ids = [] # To mark feedback entries as processed

    try:
        if retrain_on_feedback_only:
            print("Starting model retraining task based on DEDICATED FEEDBACK entries...")
            feedback_entries = crud.get_feedback_for_retraining(db, limit=500) # Get pending feedback
            if not feedback_entries:
                print("No new dedicated feedback entries found for retraining.")
                return "No new dedicated feedback entries."
            
            for entry in feedback_entries:
                texts_for_retraining.append(entry.extracted_text_at_feedback)
                labels_for_retraining.append(entry.corrected_classification)
                processed_feedback_ids.append(entry.id)
            source_description = f"{len(feedback_entries)} dedicated feedback entries"
        else:
            print("Starting model retraining task based on ALL CLASSIFIED/CORRECTED job data...")
            # This uses the existing logic to get all jobs that might have been corrected.
            # ClassificationJob.classification_result stores the latest label (original or corrected by feedback)
            # ClassificationJob.extracted_text stores the text used for that job.
            relevant_jobs = db.query(models.ClassificationJob).filter(
                models.ClassificationJob.status == JobStatus.COMPLETED, 
                models.ClassificationJob.classification_result.isnot(None),
                models.ClassificationJob.extracted_text.isnot(None)
            ).order_by(models.ClassificationJob.updated_at.desc()).limit(1000).all() # Fetch more data if retraining on all
            
            if not relevant_jobs:
                print("No relevant job data with extracted text found for retraining.")
                return "No relevant job data with text."

            for job in relevant_jobs:
                texts_for_retraining.append(job.extracted_text)
                labels_for_retraining.append(job.classification_result) 
            source_description = f"{len(relevant_jobs)} classified/corrected jobs"

        if not texts_for_retraining or not labels_for_retraining:
            print("No usable text/label pairs extracted for retraining.")
            return "No usable data for retraining."

        print(f"Retraining model with {len(texts_for_retraining)} samples from {source_description}.")
        classifier_instance.train(texts_for_retraining, labels_for_retraining, is_initial_training=False)
        
        # If we used dedicated feedback, mark it as processed
        if retrain_on_feedback_only and processed_feedback_ids:
            num_marked = crud.mark_feedback_processed(db, feedback_ids=processed_feedback_ids, status=JobStatus.COMPLETED)
            print(f"Marked {num_marked} feedback entries as processed for retraining.")
        
        print("Model retraining task completed.")
        return f"Model retrained with {len(texts_for_retraining)} samples from {source_description}."

    except Exception as e:
        print(f"Error during model retraining task: {e}")
        # Handle error, maybe log it
        return f"Error: {e}"
    finally:
        db.close() 

@celery_app.task(name="tasks.add_industry_and_retrain")
def add_industry_and_retrain_task(industry_name: str, num_documents: int):
    """ 
    Celery task to: 
    1. Generate synthetic data for the new industry.
    2. Read this data.
    3. Retrain the main classifier instance incrementally.
    """
    print(f"[Task add_industry_and_retrain] Starting for industry: {industry_name}, num_docs: {num_documents}")
    
    from src.classifier.synthetic_data import generate_and_store_synthetic_data_for_industry
    from src.classifier.ml_model import classifier_instance
    from src.config import settings # To potentially update known categories if dynamic

    try:
        # 1. Generate and store synthetic data
        # The label for these documents will be `industry_name`
        generated_files_and_labels = generate_and_store_synthetic_data_for_industry(
            industry_name=industry_name, 
            num_documents=num_documents
        )

        if not generated_files_and_labels:
            print(f"[Task add_industry_and_retrain] No synthetic data generated for {industry_name}. Retraining skipped.")
            # Optionally update a status in DB if this task had a job ID associated with it
            return f"Failed: No synthetic data generated for {industry_name}."

        texts_for_retraining = []
        labels_for_retraining = []

        # 2. Read the generated data
        for file_path, label in generated_files_and_labels:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    texts_for_retraining.append(f.read())
                    labels_for_retraining.append(label)
            except Exception as e:
                print(f"[Task add_industry_and_retrain] Error reading synthetic file {file_path}: {e}")
                # Decide: skip this file or fail the task?
                # For now, skip and continue.

        if not texts_for_retraining:
            print(f"[Task add_industry_and_retrain] Failed to read any generated synthetic data for {industry_name}. Retraining skipped.")
            return f"Failed: Could not read any synthetic data for {industry_name}."

        # Important: Update the global list of categories if the new industry is truly new
        # This ensures the model's `classes_` attribute is correctly managed during partial_fit.
        # The `classifier_instance.train` method already handles merging with existing classes.
        # However, if `settings.INITIAL_CATEGORIES` is considered the canonical list, it might need an update strategy.
        # For this setup, ml_model.py's train() dynamically adds new labels from y_train to model.classes_.
        # So, explicitly adding `industry_name` to `settings.INITIAL_CATEGORIES` at runtime is not strictly necessary
        # for the model to learn it, but `settings.INITIAL_CATEGORIES` might be used elsewhere as the definitive list.
        # Let's assume for now that `ml_model.train` handles new classes correctly.
        if industry_name not in settings.INITIAL_CATEGORIES:
             print(f"[Task add_industry_and_retrain] New industry '{industry_name}' detected. Model will learn it.")
             # If we wanted to persist this change to config (not usually done at runtime):
             # settings.INITIAL_CATEGORIES.append(industry_name)

        # 3. Retrain the model incrementally
        print(f"[Task add_industry_and_retrain] Retraining model with {len(texts_for_retraining)} new synthetic samples for {industry_name}.")
        classifier_instance.train(texts_for_retraining, labels_for_retraining, is_initial_training=False)
        
        print(f"[Task add_industry_and_retrain] Successfully processed and retrained for new industry: {industry_name}.")
        return f"Success: Added industry {industry_name} and retrained model."

    except Exception as e:
        print(f"[Task add_industry_and_retrain] Error processing new industry {industry_name}: {e}")
        # self.retry(exc=e, countdown=300, max_retries=2) # Optional retry logic
        return f"Failed: Error during processing of new industry {industry_name} - {str(e)}" 