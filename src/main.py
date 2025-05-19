# FastAPI application entry point will be defined here. 
from fastapi import FastAPI
from src.api import endpoints as api_endpoints
from src.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    # You can add more metadata like description, contact, etc.
    # docs_url="/docs", # Default is /docs
    # redoc_url="/redoc" # Default is /redoc
)

# Include API routers
app.include_router(api_endpoints.router, prefix="/api", tags=["Classification Service"])

@app.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "ok", "message": f"{settings.PROJECT_NAME} is running!"}

# To run the app (for development):
# uvicorn src.main:app --reload

# Note on database creation and initial model training:
# The on_startup event in api/endpoints.py handles:
# 1. Creation of database tables via `create_db_tables()` from `src.db.database`.
# 2. Initial training of the ML model via `train_initial_model_from_scratch()` from `src.classifier.ml_model` if no pre-trained model is found.

# To run Celery worker (in a separate terminal):
# celery -A src.celery_tasks.celery_app worker -l info -P eventlet # Use -P eventlet or gevent on Windows 

# To run redis server (in a separate terminal):
# redis-server