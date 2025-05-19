# Heron AI File Classifier
[![Tests](https://github.com/beubax/join-the-siege/actions/workflows/test.yml/badge.svg)](https://github.com/beubax/join-the-siege/tests)

This project provides a robust and extensible system for classifying documents into predefined industry categories based on their text content. It is designed for scalability, adaptability to new document types and industries, and efficient processing of large document volumes. I roughly spent ~6 hours on this challenge today (May 19th, 2025).

## Core Features

*   **Content-Based Classification**: Accurately classifies documents by extracting and analyzing their full text content, rather than relying on potentially misleading filenames.
*   **Hybrid ML Approach for Accuracy & Efficiency**:
    *   **Lightweight Primary Classifier**: Employs an efficient, classic machine learning model (configurable, e.g., SGDClassifier, MultinomialNB) for initial classification. This model is quick, deterministic, and inexpensive to run.
    *   **Transformer Escalation**: When the primary lightweight model's confidence in a prediction is below a configurable threshold, the system can escalate to a more powerful (though heavier) on-device transformer model for a more nuanced classification. This hybrid approach balances speed with accuracy, achieving significant performance gains (e.g., tested accuracy improved from ~61% to ~79% with minimal initial training data and few escalations).
    *   **Privacy-Focused**: Document content, including escalations to the on-device transformer, remains within the local environment, ensuring data privacy.
*   **Automated Synthetic Data Generation**:
    *   **Cold Starts**: For initial setup and to bootstrap the model, synthetic documents are automatically generated for predefined industries using (simulated) Large Language Model (LLM) calls.
    *   **New Industries**: When a new industry is added via the API, the system automatically generates tailored synthetic training data for that industry.
*   **Incremental Model Updates & Continuous Learning**:
    *   **Feedback Loop**: Supports a human-in-the-loop process where corrected classifications can be submitted for documents.
    *   **`partial_fit`**: The underlying ML models leverage `partial_fit` (online learning), allowing them to be incrementally updated with new data from feedback or newly added industries without requiring complete retraining from scratch. This is resource-efficient and enables rapid adaptation to evolving data patterns and new categories.
*   **Asynchronous Task Processing**: Leverages Celery and Redis for background processing of computationally intensive tasks like file parsing, ML inference, synthetic data generation, and model retraining. This ensures API responsiveness and enhances system scalability.
*   **Data Persistence**: Utilizes a lightweight SQLite database to store job information, extracted text content from documents, classification results, and dedicated feedback entries. Storing extracted text is crucial for effective model retraining.
*   **Configurable & Extensible**:
    *   **Swappable ML Models**: The core lightweight ML model can be easily swapped (e.g., from SGDClassifier to MultinomialNB) via a configuration setting.
    *   **Modular Design**: Built with a modular architecture (`api`, `classifier`, `db`, `celery_tasks`) for maintainability and ease of extension.
*   **Dockerized for Portability**: Includes `Dockerfile` and `docker-compose.yml` for easy, consistent deployment across different environments.
*   **Testing Foundation**: Core functionalities are covered by unit tests using `pytest`, providing a basis for robust development.

## How to Use

### Prerequisites
*   Python 3.10+
*   Redis server
*   Docker and Docker Compose (for containerized deployment)
*   OpenAI API Key (For synthetic data simulation. The data is, by default generated and stored in the github repo, however LLM call is needed if trying to add new industries. Store in `.env` file as `OPENAI_API_KEY="your_key_here"`).

### Option 1: Local Development Environment

1.  **Setup**:
    ```bash
    # git clone <repository_url> && cd <repository_name> # If not already cloned
    python3 -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install --upgrade pip
    pip install -r requirements.txt
    # Create/update .env file in the project root, especially for OPENAI_API_KEY.
    # Example: echo "OPENAI_API_KEY='your_openai_api_key_here'" > .env 
    ```

2.  **Run Services**:
    *   **Redis Server**: Ensure your Redis server is running. If installed locally, it might start automatically or require a command like `redis-server` or `sudo systemctl start redis`. (Alternatively, for an isolated Redis, use `docker run -d -p 6379:6379 --name heron-redis redis:alpine`)
    *   **FastAPI Server (Terminal 1)**:
        ```bash
        uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
        ```
        The API will be available at `http://127.0.0.1:8000`. Swagger UI documentation at `http://127.0.0.1:8000/docs`.
    *   **Celery Worker (Terminal 2)**:
        ```bash
        celery -A src.core.celery_app worker -l info
        ```

### Option 2: Docker & Docker Compose

1.  **Setup**: Ensure an `.env` file is present in the project root if you need to set environment variables like `OPENAI_API_KEY`.
2.  **Build and Run**:
    ```bash
    docker compose up --build
    ```
    (Use `-d` for detached mode. View logs with `docker-compose logs -f app` or `docker-compose logs -f celery_worker`.)

## API Endpoints

The API is served at `http://127.0.0.1:8000` (or your Docker host). Interactive API documentation (Swagger UI) is available at `http://127.0.0.1:8000/docs`.

**Base URL**: `/api` (e.g., `http://127.0.0.1:8000/api`)

*   `POST /classify_file/`: Upload a file (`multipart/form-data`, field name `file`) for classification. The system queues the file for processing and returns a job ID.
*   `GET /jobs/{job_id}`: Retrieve the status (e.g., PENDING, PROCESSING, COMPLETED, FAILED) and, if completed, the classification result for a given job ID.
*   `POST /feedback/`: Submit a corrected classification for a previously processed job. This stores the feedback in a dedicated table and updates the original job's classification for immediate reflection.
    *   Payload: `{"job_id": "your_job_id", "corrected_classification": "correct_industry_label"}`
*   `POST /industries/`: Add a new industry to the system. This triggers background tasks to generate synthetic training data for this new industry and then incrementally retrains the ML model.
    *   Payload: `{"industry_name": "new_industry_name", "num_synthetic_documents": 10}`
*   `POST /retrain_model/`: Manually trigger model retraining. Allows choosing the data source for retraining.
    *   Payload: `{"retrain_on_feedback_only": false}` (Set to `true` to use only dedicated feedback entries; `false` uses all relevant classified/corrected job data).
*   `GET /health`: Basic application health check.

## Design Choices & Rationale

This system is engineered with a focus on practicality, adaptability, and real-world ML considerations:

*   **Lightweight ML Core for Speed and Cost-Effectiveness**: The primary classification engine is a classic machine learning model (e.g., SGDClassifier, MultinomialNB, configurable via `ML_MODEL_TYPE`). These models are chosen for their:
    *   **Efficiency**: They are fast to train and quick for inference, making them suitable for handling document volumes.
    *   **Determinism**: Given the same input and model state, they produce the same output, aiding in predictability and debugging.
    *   **Incremental Learning**: Crucially, they support `partial_fit`, enabling the model to learn from new data (feedback, new industries) on the fly without costly full retraining cycles. This helps the model adapt to real-world data drift and new patterns efficiently.
    *   **Resource Friendliness**: They are generally less resource-intensive than large neural networks, reducing operational costs.

*   **Transformer Escalation for Enhanced Accuracy**: While lightweight models handle the bulk of classifications, the system incorporates an escalation path. If the primary model's confidence for a prediction falls below a configurable threshold (`PREDICTION_CONFIDENCE_THRESHOLD`), the task can be escalated to a more sophisticated on-device transformer model. This provides a significant accuracy boost for ambiguous cases.
    *   **Performance Gains**: Initial tests showed that this hybrid approach can substantially improve classification accuracy (e.g., from ~61% to ~75%) even with a small initial training set for the lightweight model and a limited number of escalations.
    *   **Privacy by Design**: The escalation transformer runs on-device (CPU-only), ensuring that document content never leaves the platform, addressing privacy concerns. For even more robust performance in less sensitive contexts, future extensions could consider calls to external, state-of-the-art LLMs.

*   **The Power of the Feedback Loop**: Real-world data rarely perfectly matches synthetic or initial training data. The feedback mechanism is vital:
    *   **Targeted Improvement**: By allowing users to submit corrections, the system gathers examples where the current model errs.
    *   **`partial_fit` in Action**: When retraining is triggered (either on dedicated feedback or all data), these corrected samples, along with their original text (which is stored in the database), are used to update the model via `partial_fit`. This method adjusts the model weights based on the new information without "forgetting" what it learned from older data (mitigating catastrophic forgetting to a large extent, especially when class distributions are somewhat stable or new data isn't overwhelmingly skewed). This iterative refinement is key to a production ML system that improves over time.

*   **Synthetic Data for Agility**: The ability to generate synthetic data (currently simulated LLM calls, but extendable to real ones) is crucial for:
    *   **Bootstrapping**: Quickly creating an initial training set when no real data is available.
    *   **New Industries**: Rapidly adapting the model when new classification categories are introduced, ensuring the model has examples to learn from for these new classes.

*   **Systematic Model Evaluation**: Each time the initial model is trained (e.g., on startup or when `train_initial_model_from_scratch` is invoked), it undergoes a validation step. Data is split into training and testing sets (70-30), and metrics such as overall accuracy and a detailed per-class classification report (including precision, recall, F1-score) are logged. This provides insights into model performance and areas for improvement.

*   **FastAPI & Asynchronous Processing**:
    *   **FastAPI**: Chosen for its high performance (ASGI), built-in data validation with Pydantic, and automatic OpenAPI/Swagger documentation, accelerating development.
    *   **Celery & Redis**: Document processing, ML inference, synthetic data generation, and model retraining are offloaded to Celery background workers, with Redis as the message broker. This keeps the API responsive and allows the system to handle long-running tasks and scale effectively.

*   **SQLite for Simplicity & Portability**: SQLite is used for data persistence (job details, extracted text, feedback). It's lightweight, file-based, and suitable for the project's scope, while SQLAlchemy ORM allows for potential future migration to other SQL databases. Storing the extracted text directly is a key choice, ensuring that the exact content used for an initial prediction is available for reliable retraining based on feedback.

## Project Structure

```
heron_classifier/
├── .env                    # Environment variables (e.g., API keys)
├── .gitignore
├── Dockerfile              # For building the application image
├── docker-compose.yml      # For running all services (app, worker, redis)
├── README.md               # This README file
├── requirements.txt        # Python dependencies
├── data/                   # Data directory (SQLite DB, ML models, synthetic data folder)
│   ├── classifier.db       # SQLite database file
│   ├── classifier_model.joblib # Serialized trained ML model
│   ├── vectorizer.joblib   # Serialized text vectorizer
│   └── synthetic_data/     # Stores generated synthetic documents (by industry)
├── src/                    # Main application source code
│   ├── __init__.py
│   ├── api/                # FastAPI endpoints, Pydantic schemas
│   ├── classifier/         # File parsing, ML model logic, synthetic data generation
│   ├── db/                 # Database models, CRUD operations, session management
│   ├── celery_tasks/       # Celery app worker and background task definitions
│   └── main.py             # FastAPI application entry point
|   └── config.py           # Configuration for entire app
└── tests/                  # Pytest unit and integration tests
    ├── __init__.py
    ├── api/
    ├── classifier/
    ├── conftest.py         # Shared test fixtures and configuration
    ├── core/
    ├── db/
    └── tasks/
```

## Running Tests

Ensure the development environment is activated (if not using Docker for tests) and dependencies are installed:
```bash
pytest -v tests
```
The tests cover core features and utilize an in-memory SQLite database to avoid interference with development data.

## Potential Future Enhancements

This platform provides a solid foundation, but there's always room for growth. Key areas for future development include:

*   **Advanced File Handling**:
    *   **Expanded File Types**: Incorporate robust parsing for a wider array of file formats (e.g., Excel spreadsheets, older `.doc` files, image-based PDFs with OCR).
    *   **Structured Data Extraction**: Improve extraction logic to preserve and utilize the structure within documents, such as tables in PDFs or spreadsheets, which can be crucial for certain classification tasks.
*   **Refined ML & NLP Capabilities**:
    *   **Full LLM Integration**: Transition synthetic data generation from simulation to actual API calls to services like OpenAI GPT-4o or other leading LLMs for higher quality and more diverse training data.
    *   **Smarter Escalation**: Implement more sophisticated logic for when and how to escalate to the transformer model, potentially based on specific error patterns or confidence distributions.
    *   **Alternative Models**: Experiment with other lightweight ML models or even fine-tuning smaller, specialized transformer models for the primary classification task.
    *   **Vector Databases**: For very large-scale feedback or document corpora, consider using vector databases for efficient similarity searches related to feedback or for finding analogous documents.
*   **Enhanced Feedback System**:
    *   **Granular Feedback**: Allow feedback on specific text segments within a document, not just the overall classification.
    *   **Feedback Review Workflow**: Implement a UI or process for reviewing and prioritizing feedback before it's used for retraining.
*   **Operational Excellence**:
    *   **Comprehensive Testing**: While core features are tested, expand test coverage significantly, especially for edge cases, integration points, and potential failure modes. A dedicated effort here is invaluable.
    *   **Production Monitoring & Logging**: Integrate comprehensive structured logging (e.g., ELK stack, Grafana Loki) and application performance monitoring (APM) tools for better observability in production.
    *   **Database Migrations**: Introduce a database migration tool (e.g., Alembic) for managing schema changes systematically in production environments.
    *   **Security Hardening**: Implement robust API authentication and authorization layers, input sanitization, and other security best practices.
*   **Cloud Deployment & Scalability Enhancements**:
    *   **Decoupled File Handling**: For cloud deployments (e.g., Kubernetes, AWS ECS/Fargate, Google Cloud Run), transition from shared host volumes (like `temp_uploads`) to cloud object storage (e.g., AWS S3, Google Cloud Storage, Azure Blob Storage) for temporary file sharing between services (API and Celery workers). The API service would upload files to object storage, and Celery workers would download them for processing.
    *   **Managed Database**: Migrate from SQLite to a managed cloud database service (e.g., AWS RDS, Google Cloud SQL, Azure Database for PostgreSQL/MySQL) for production-grade data persistence, scalability, and reliability.
    *   **Container Orchestration**: Develop deployment configurations (e.g., Helm charts for Kubernetes, CDK/CloudFormation/Terraform scripts) for robust cloud deployment.
    *   **Stateless Services**: Ensure API and worker services are designed to be as stateless as possible, relying on external services (object storage, databases, message queues) for state management to facilitate easier scaling and resilience.
*   **Dynamic Category Management**: Shift management of industry categories from the current configuration list to a dynamic source, such as a dedicated database table, allowing for easier management via an admin interface.
