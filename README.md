# MNIST ANN Classifier — Full-Stack ML Project

A production-style end-to-end project: ANN model trained on MNIST → served via FastAPI → interactive Streamlit frontend.

## Project Structure

```
mnist-ann-project/
├── api/                    # FastAPI backend
│   ├── app/
│   │   ├── main.py         # App entry point
│   │   ├── config.py       # Settings / env vars
│   │   ├── routers/        # Route handlers
│   │   ├── schemas/        # Pydantic models
│   │   ├── services/       # Business logic (model inference)
│   │   └── utils/          # Helpers (image processing)
│   ├── tests/              # Pytest unit tests
│   └── requirements.txt
├── frontend/               # Streamlit app
│   ├── app.py              # Main Streamlit entry
│   ├── pages/              # Multi-page support
│   ├── components/         # Reusable UI components
│   ├── utils/              # API client, helpers
│   └── requirements.txt
├── model/                  # Saved model artifacts
│   └── model.h5            ← place your downloaded model here
├── notebooks/              # Jupyter exploration (reference)
├── scripts/                # Utility scripts
├── docker-compose.yml      # Run everything together
├── .env.example            # Environment variable template
└── README.md
```

## Quick Start

### 1. Place Your Model
```bash
cp /path/to/your/model.h5 model/model.h5
```

### 2. Setup Environment
```bash
cp .env.example .env
```

### 3. Run with Docker (Recommended)
```bash
docker-compose up --build
```

- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Frontend: http://localhost:8501

### 4. Run Locally (Without Docker)
```bash
# Terminal 1 — API
cd api
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/v1/model/info` | Model metadata |
| POST | `/api/v1/predict` | Predict digit from image |
| POST | `/api/v1/predict/batch` | Predict multiple images |

## Tech Stack
- **Model**: TensorFlow/Keras ANN
- **API**: FastAPI + Uvicorn
- **Frontend**: Streamlit
- **Containerization**: Docker + Docker Compose
