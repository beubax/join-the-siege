from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from src.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    # For SQLite, connect_args is needed to enable foreign key constraints if we use them
    # and to allow multiple threads to share the same connection (important for FastAPI)
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Function to create database tables
# This should be called once at application startup (e.g., in main.py or a startup script)
# For production, you'd typically use Alembic migrations.
def create_db_tables():
    # Import all modules here that define models so that
    # they are registered with Base.metadata
    # from . import models # This line would cause circular import if models.py imports Base from here
    Base.metadata.create_all(bind=engine) 