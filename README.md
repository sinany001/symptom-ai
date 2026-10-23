# SymptomAI Agent

## Project Overview

SymptomAI is a symptom information and risk assessment application built with Python and FastAPI. It includes an existing machine learning prediction pipeline and a local AI agent for extracting structured information from user-reported symptoms.

The AI agent uses Ollama with the `llama3.2:1b` model to organize symptom descriptions into structured JSON.

**Important:** This project is an educational tool and is not a substitute for professional medical advice, diagnosis, or treatment.

## Features

- FastAPI REST API
- User registration and login
- Authenticated symptom prediction
- Symptom extraction and risk assessment
- Local AI-powered symptom information extraction
- Structured JSON responses
- Automated tests with pytest

## Technologies

- Python 3.11
- FastAPI
- Pydantic
- Ollama
- Llama 3.2 1B
- pytest
- Existing machine learning model and TF-IDF vectorizer
- PostgreSQL

## Prerequisites

- Python 3.11
- Ollama
- The project’s required Python dependencies
- PostgreSQL configured for the existing application

## Installation

### 1. Clone or open the project

Open a terminal in the project directory.

### 2. Create and activate a virtual environment

On macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

Install the packages required by the project. If a `requirements.txt` file is available, use:

```bash
python -m pip install -r requirements.txt
```

The Ollama Python package can be installed with:

```bash
python -m pip install ollama
```

Install any other dependencies required by the existing application.

### 4. Install and start Ollama

Download Ollama from https://ollama.com/download.

Download the model:

```bash
ollama pull llama3.2:1b
```

Verify the installed model:

```bash
ollama list
```

### 5. Configure the existing application

Configure the database and authentication settings required by `main.py`. Keep secrets in environment variables and do not commit them to version control.

## Running the Application

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

Open the API documentation:

http://127.0.0.1:8000/docs

## API Endpoints

### `POST /agent/extract`

Extracts structured information from a user-provided symptom description.

Example request:

```json
{
  "symptoms": "I have had a cough and fever for two days."
}
```

Example response:

```json
{
  "symptoms": ["cough", "fever"],
  "duration": "2 days",
  "severity": "unknown",
  "missing_information": [],
  "follow_up_questions": [],
  "uncertainty_notes": []
}
```

The actual response depends on the model output. The example illustrates the response structure, not a guaranteed result.

This endpoint requires authentication.

### `POST /predict`

Uses the existing prediction pipeline to process symptom input and return the application's prediction and risk assessment.

Authentication is required.

### Other endpoints

The application also includes registration, login, user information, and analysis-history endpoints. See `/docs` for the current API schema.

## Running Tests

Run all tests:

```bash
python -m pytest -v
```

Run the AI agent tests only:

```bash
python -m pytest tests/test_ai_agent.py -v
```

## Limitations and Safety

- The local language model can produce incorrect or inconsistent results.
- Extracted information should be checked against the original user input.
- The model must not be treated as a medical diagnostic system.
- The severity field should remain unknown unless the user explicitly provides enough information.
- Do not use model output alone to make medical decisions.
- Protect authentication tokens, database credentials, and other sensitive information.

## Project Structure

```text
symptom-ai/
├── main.py
├── ai_agent.py
├── symptom_extractor.py
├── risk_assessment.py
├── models/
├── tests/
├── data/
├── frontend/
└── README.md
```

Additional files may be present depending on the local development environment.

## License

Add the project's license information here if applicable.
