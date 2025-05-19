# Heron Data Challenge
[![Tests](https://github.com/beubax/join-the-siege/actions/workflows/test.yml/badge.svg)](https://github.com/beubax/join-the-siege/tests)

This project provides a system for classifying documents into predefined industry categories based on their text content. I roughly spent ~6 hours on this challenge today (May 19th, 2025).

## Core Features

*   **Intelligent Document Understanding**: Accurately classifies documents based on their actual text content, not just filenames.
*   **Hybrid ML Approach for Optimal Performance**:
    *   Employs a swift, lightweight primary model for rapid, everyday classifications.
    *   When faced with ambiguity, it intelligently escalates to a more powerful on-device transformer model. This boosts accuracy considerably while keeping all document data private.
*   **Automated Training Data Generation**: Utilizes LLM capabilities to automatically create synthetic documents, essential for cold starts and for integrating new industry categories.
*   **Continuous Improvement via Feedback Loop**: The system is designed to learn and adapt. User corrections on classifications are captured and used to incrementally refine the ML model (via `partial_fit`), making it smarter over time without requiring costly complete retraining.
*   **Efficient Background Processing**: Leverages Celery and Redis to handle demanding tasks like file analysis, ML inference, and model updates asynchronously, ensuring the API remains responsive and the system can scale.
*   **Data-Driven Model Evolution**: Persistently stores job statuses, extracted text and results (in SQLite), providing a rich dataset for ongoing model improvement and learning from historical patterns.
*   **Flexible and Extensible Architecture**: Features a modular design and allows for easy substitution of core components (like the primary ML model), facilitating customization and future development.
*   **Simplified Deployment with Docker**: Includes `Dockerfile` and `docker-compose.yml` for straightforward, consistent setup across various environments.
*   **Reliability Assured by Testing**: Core functionalities are validated through `pytest` unit tests, providing a dependable foundation. Tests run automatically via github actions when pushed to the main branch.

## How to Use

### Prerequisites
*   Python 3.10+
*   Redis server
*   Docker and Docker Compose (for containerized deployment)
*   OpenAI API Key (For synthetic data generation. The data is, by default generated and stored in the github repo, however LLM call is needed if trying to add new industries. Store in `.env` file as `OPENAI_API_KEY="your_key_here"`).

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

This section outlines some of the thinking behind the system's architecture and component choices. The goal was to build a practical and adaptable classification tool.

*   **Primary Classifier: Lightweight for Efficiency**:
    For the main classification task, I opted for a classic machine learning model (e.g., SGDClassifier, configurable via `ML_MODEL_TYPE`). These models offer a good balance between:
    *   **Speed**: They are generally quick for both training and making predictions, which is helpful when dealing with many documents.
    *   **Incremental Learning**: A key feature is their support for `partial_fit`. This allows the model to learn from new data (like user feedback or documents from a new industry) incrementally, without needing to be retrained from scratch each time. This helps the model adapt to real-world data drift and new patterns efficiently.
    *   **Determinism**: Given the same input and model state, they produce the same output, aiding in predictability and debugging.

*   **Transformer Escalation:**:
    While the lightweight model handles most cases, I included an option to escalate to an on-device transformer model. This happens if the primary model's confidence in a prediction is low (below `PREDICTION_CONFIDENCE_THRESHOLD`).
    *   **Improved Accuracy for Tough Cases**: Transformers can offer more nuanced understanding for complex or ambiguous documents, leading to better accuracy in those specific instances. Initial tests showed that this hybrid approach can improve classification accuracy (e.g., from ~61% to ~75%) even with a small initial training set and a limited number of escalations
    *   **Local Processing for Privacy**: By running the transformer on-device, document content remains within the local environment, which can be important for privacy considerations. 

*   **Feedback Loop: Enabling Continuous Improvement**:
    Real-world data rarely perfectly matches synthetic or initial training data. The system includes a feedback mechanism:
    *   **Learning from Corrections**: Users can submit corrected classifications. This feedback, along with the original document text (which I store), is then used to update the primary model using `partial_fit`.
    *   **Iterative Refinement**: This method adjusts the model weights based on the new information without "forgetting" what it learned from older data (mitigating catastrophic forgetting to a large extent, especially when class distributions are somewhat stable or new data isn't overwhelmingly skewed). This iterative refinement is key to a production ML system that improves over time.

*   **Synthetic Data:**:
    To get the system started, especially when real labeled data might be scarce, I incorporated a way to generate synthetic documents (currently using an LLM - gpt4o).
    *   **Bootstrapping**: This helps create an initial training set.
    *   **Adapting to New Categories**: When a new industry is added, synthetic data generation provides initial examples for that category, allowing the model to start learning about it quickly.

*   **Model Evaluation: Understanding Performance**:
    Whenever the primary model is trained or retrained from a larger dataset, I perform a validation step. The data is split, and I log metrics like overall accuracy and a detailed per-class classification report (including precision, recall, F1-score). This gives us a snapshot of how well the model is performing and where it might be struggling.

*   **FastAPI & Asynchronous Tasks: For a Responsive System**:
    *   **FastAPI**: I chose FastAPI for the API layer due to its performance, built-in data validation (Pydantic), and automatic generation of interactive API documentation, which speeds up development.
    *   **Celery & Redis**: Tasks that can take time—like parsing files, running ML models, generating synthetic data, or retraining—are handled by Celery background workers, with Redis as the message broker. This keeps the main API responsive, as it doesn't have to wait for these longer operations to complete.

*   **SQLite: Simple and Practical Data Storage**:
    For storing job information, extracted text, and feedback, I use SQLite. It's a lightweight, file-based database that's easy to set up and manage for a project of this nature. Storing the extracted text itself is important, as it ensures the exact content used for a prediction is available for reliable retraining if feedback is provided. The use of SQLAlchemy ORM also offers a path to potentially switch to a different SQL database in the future if needed.

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

While the current system is robust, several areas offer opportunities for future development:

*   **Advanced File Handling**:
    *   Support a broader range of file types (e.g., Excel, older `.doc`, image-based PDFs via OCR).
    *   Improve extraction to leverage structured content like tables within documents.
*   **Refined ML & NLP Capabilities**:
    *   Develop more sophisticated criteria for escalating to the transformer model.
    *   Explore alternative lightweight or fine-tuned transformer models for primary classification.
    *   Consider vector databases for managing very large document or feedback corpora.
*   **Enhanced Feedback System**:
    *   Allow feedback on specific text segments, not just overall document classification.
    *   Implement a review workflow for submitted feedback before retraining.
*   **Operational Excellence**:
    *   Expand test coverage comprehensively, especially for integrations and edge cases.
    *   Integrate production-grade structured logging and application performance monitoring (APM).
    *   Introduce database migration tools (e.g., Alembic) for schema management.
    *   Bolster security with robust API authentication, authorization, and input sanitization.
*   **Cloud Deployment & Scalability**:
    *   Utilize cloud object storage (S3, GCS) for temporary file handling in distributed environments.
    *   Migrate from SQLite to a managed cloud database service (RDS, Cloud SQL) for production.
    *   Develop configurations for container orchestration (e.g., Kubernetes Helm charts).
    *   Further ensure services are stateless for easier scaling and resilience.
*   **Dynamic Category Management**: Shift industry category management from configuration files to a dynamic database source, enabling easier updates via an admin interface.
