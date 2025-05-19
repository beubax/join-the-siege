from celery import Celery
from src.config import settings

celery_app = Celery(
    "heron_classifier_tasks", # Application name
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=['src.celery_tasks.tasks'] # List of modules to import when worker starts
)

# Optional Celery configuration
celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    # Optional: set a default task execution time limit
    # task_time_limit=300, # 5 minutes
    # Optional: set a default task retry policy
    # task_acks_late=True,
    # worker_prefetch_multiplier=1, # If tasks are long-running
)

if __name__ == '__main__':
    # This allows running celery worker directly using:
    # python -m src.celery_tasks.celery_app worker -l info
    # (Or more commonly, `celery -A src.celery_tasks.celery_app worker -l info` from project root)
    celery_app.start() 