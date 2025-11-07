# Shruti Parser

## Overview

Shruti Parser is a sophisticated LangChain-based text analysis microservice built with FastAPI. It leverages the Groq API (powered by Llama 3.1-8B Instant model) to perform intelligent text analysis tasks including intent classification, action phrase extraction, and text summarization. The service is designed as a microservice architecture, making it easily integrable into larger applications for natural language processing needs.

### What It Does

The service analyzes input text and provides three key outputs:
- **Intent Classification**: Categorizes text into one of four intents: `command` (instructions to perform actions), `query` (questions seeking information), `teaching` (explanatory or instructional content), or `data` (factual information or measurements).
- **Action Extraction**: Identifies and extracts key action verbs or phrases from the text, useful for understanding actionable steps.
- **Summarization**: Generates concise, one-sentence summaries of the input text.

### Key Features

- **LangChain Integration**: Utilizes LangChain's prompt templates and output parsers for structured AI interactions.
- **Groq API Backend**: Employs Groq's fast inference API for cost-effective and efficient LLM calls.
- **Asynchronous Processing**: Built with async/await patterns for non-blocking API calls.
- **Robust Error Handling**: Comprehensive logging and error management with fallback mechanisms.
- **RESTful API**: Clean FastAPI endpoints with Pydantic validation.
- **Comprehensive Testing**: Unit tests with sample data covering various text types.
- **Environment-Based Configuration**: Secure API key management via environment variables.

### Technology Stack

- **Framework**: FastAPI for REST API development
- **AI/ML**: LangChain for prompt engineering, Groq API for LLM inference
- **Language**: Python 3.8+
- **Validation**: Pydantic for request/response models
- **Configuration**: python-dotenv for environment variable loading
- **Testing**: unittest framework with requests for API testing

## Setup

### Prerequisites

- Python 3.8 or higher
- A Groq API account and API key (sign up at https://console.groq.com)

### Installation Steps

1. **Clone or download the project**:
   ```bash
   # If cloning from repository
   git clone <repository-url>
   cd Shruti-Parser
   ```

2. **Create a virtual environment**:
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file in the project root:
   ```bash
   touch .env  # On Windows: type nul > .env
   ```

   Add your Groq API key to the `.env` file:
   ```
   GROQ_API_KEY=your_actual_groq_api_key_here
   ```

   **Security Note**: Never commit the `.env` file to version control. It's already included in `.gitignore` if present.

5. **Verify installation**:
   ```bash
   python -c "import fastapi, groq, langchain_core; print('All dependencies installed successfully')"
   ```

## Running the Service

### Development Mode

Start the FastAPI server with auto-reload for development:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

For production deployment, use a production ASGI server:
```bash
# Using uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4

# Or using gunicorn with uvicorn workers
pip install gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Accessing the Service

Once running, the service will be available at:
- **Base URL**: http://localhost:8000
- **Interactive API Docs**: http://localhost:8000/docs (Swagger UI)
- **Alternative API Docs**: http://localhost:8000/redoc (ReDoc)

### Health Check

Test if the service is running:
```bash
curl http://localhost:8000/docs
```
You should see the FastAPI interactive documentation page.

## API Endpoints

The service provides three main endpoints, all accepting POST requests with JSON payloads containing a `text` field.

### Request Schema

All endpoints accept:
```json
{
  "text": "string (required, non-empty)"
}
```

### 1. POST /analyze - Complete Text Analysis

**Description**: Performs comprehensive text analysis returning intent classification, action extraction, and summarization in a single response.

**Response Schema**:
```json
{
  "intent": "string (one of: command, query, teaching, data)",
  "actions": ["string"],
  "summary": "string"
}
```

**Example Request**:
```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "Write a poem about sunset"}'
```

**Example Response**:
```json
{
  "intent": "command",
  "actions": ["write poem", "describe sunset"],
  "summary": "The user requests creation of a poem describing a sunset."
}
```

### 2. POST /extract - Action Phrase Extraction

**Description**: Extracts key action verbs or phrases from the input text.

**Response Schema**:
```json
{
  "actions": ["string"]
}
```

**Example Request**:
```bash
curl -X POST "http://localhost:8000/extract" \
     -H "Content-Type: application/json" \
     -d '{"text": "First, preheat the oven to 350°F. Then mix flour, sugar, and eggs."}'
```

**Example Response**:
```json
{
  "actions": ["preheat oven", "mix ingredients"]
}
```

### 3. POST /summarize - Text Summarization

**Description**: Generates a concise, one-sentence summary of the input text.

**Response Schema**:
```json
{
  "summary": "string"
}
```

**Example Request**:
```bash
curl -X POST "http://localhost:8000/summarize" \
     -H "Content-Type: application/json" \
     -d '{"text": "The average temperature in New York last week was 72°F with 65% humidity."}'
```

**Example Response**:
```json
{
  "summary": "New York experienced an average temperature of 72°F and 65% humidity last week."
}
```

### Error Responses

All endpoints return appropriate HTTP status codes and error messages:

- **400 Bad Request**: Empty or invalid input text
- **500 Internal Server Error**: Processing failures (with detailed error messages)

**Error Response Schema**:
```json
{
  "detail": "string"
}
```

## Architecture & How It Works

### Core Components

The service is built around three main architectural layers:

1. **API Layer** (FastAPI)
   - RESTful endpoints with automatic request validation
   - Pydantic models for type safety
   - Asynchronous request handling

2. **Processing Layer** (LangChain + Custom Logic)
   - Prompt engineering with LangChain templates
   - Custom GroqLLM wrapper for async API calls
   - Text processing pipelines for each analysis type

3. **AI/ML Layer** (Groq API)
   - Llama 3.1-8B Instant model for inference
   - Structured prompts for consistent outputs
   - Fallback mechanisms for error handling

### Processing Flow

```
Input Text → FastAPI Endpoint → Validation → LangChain Pipeline → Groq API → Response Processing → JSON Output
```

### Key Classes and Functions

- **`GroqLLM`**: Custom LLM wrapper that handles async Groq API calls using `asyncio.to_thread()` to avoid blocking the event loop.

- **`classify_text()`**: Uses regex-based output normalization and fallback prompts to ensure reliable intent classification.

- **`extract_actions_from_text()`**: Extracts action phrases using targeted prompts and comma-separated parsing.

- **`summarize_text_content()`**: Generates one-sentence summaries with strict formatting rules.

### LangChain Integration

The service leverages LangChain for:
- **Prompt Templates**: Structured prompts for consistent AI interactions
- **Output Parsers**: Basic string parsing (could be extended to more complex parsers)
- **Runnable Components**: Modular pipeline design (though simplified in this implementation)

### Async Design

All AI calls are wrapped in async functions using `asyncio.to_thread()` to prevent blocking FastAPI's async event loop, ensuring high concurrency and responsiveness.

## Project Structure

```
Shruti-Parser/
├── app.py                    # Main FastAPI application with endpoints and core logic
├── requirements.txt          # Python dependencies with versions
├── .env                      # Environment variables (API keys) - not in version control
├── test_data/               # Test fixtures and sample data
│   └── samples.json         # JSON file with test cases covering all intent types
├── test_shruti_parser.py    # Unit tests using unittest framework
├── __pycache__/             # Python bytecode cache (auto-generated)
└── README.md                # This documentation file
```

### File Descriptions

- **`app.py`** (216 lines): Contains the entire application logic including:
  - FastAPI app initialization
  - Pydantic request/response models
  - Groq client setup and custom LLM wrapper
  - Three async processing functions (classify, extract, summarize)
  - Three API endpoints (/analyze, /extract, /summarize)
  - Comprehensive error handling and logging

- **`test_data/samples.json`**: Contains 5 diverse test samples covering:
  - Creative commands ("Write a poem about sunset")
  - Knowledge queries ("What is artificial intelligence?")
  - Instructional content ("To make a cake, first preheat...")
  - Data/facts ("The average temperature in New York...")

- **`test_shruti_parser.py`**: Unit test suite that:
  - Loads test data from samples.json
  - Tests all three endpoints against expected outputs
  - Validates response structure and content

## Testing

### Running Tests

Execute the complete test suite:
```bash
python -m unittest test_shruti_parser.py -v
```

### Test Coverage

The test suite covers:
- **Endpoint Validation**: All three endpoints (/analyze, /extract, /summarize)
- **Response Structure**: Validates JSON response formats
- **Intent Classification**: Tests against known intent categories
- **Action Extraction**: Verifies action phrase parsing
- **Summarization**: Checks summary generation quality

### Test Data

Tests use real-world examples from `test_data/samples.json`:
- Creative writing commands
- Scientific queries
- Recipe instructions
- Weather data reports

### Manual Testing

Use curl commands or the interactive API docs at `http://localhost:8000/docs` for manual endpoint testing.

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | Latest | Web framework for building REST APIs |
| `uvicorn[standard]` | Latest | ASGI server for running FastAPI applications |
| `langchain` | Latest | Framework for building LLM applications and prompt engineering |
| `groq` | Latest | Official Python client for Groq API |
| `pydantic` | Latest | Data validation and serialization |
| `python-dotenv` | Latest | Environment variable loading from .env files |
| `typing-extensions` | Latest | Enhanced type hints for older Python versions |

### Dependency Management

All dependencies are listed in `requirements.txt` and can be installed with:
```bash
pip install -r requirements.txt
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | API key for accessing Groq services. Obtain from https://console.groq.com |

### Security Notes

- The `.env` file containing API keys should never be committed to version control
- Use environment-specific `.env` files (`.env.local`, `.env.production`)
- Rotate API keys regularly for security

## Error Handling and Logging

### Logging Configuration

The application uses Python's built-in `logging` module configured at INFO level:
- Logs API initialization status
- Records all Groq API calls and responses
- Captures errors with detailed context
- Logs classification results and fallbacks

### Error Handling Strategies

1. **API Key Validation**: Checks for `GROQ_API_KEY` at startup
2. **Input Validation**: FastAPI automatically validates request schemas
3. **Empty Text Handling**: Returns 400 errors for empty inputs
4. **LLM Fallbacks**: Classification includes fallback prompts for unreliable outputs
5. **Graceful Degradation**: Defaults to "query" intent if classification fails completely

### Common Error Scenarios

- **Missing API Key**: Application fails to start with clear error message
- **Invalid API Key**: Groq client initialization fails
- **Empty Input**: HTTP 400 with "Input text is empty" message
- **LLM Errors**: HTTP 500 with detailed error descriptions
- **Network Issues**: Propagated from Groq client with context

### Monitoring

Logs provide comprehensive monitoring information:
- Request timestamps and processing times
- Success/failure rates
- Error types and frequencies
- API usage patterns

---

## Quick Start

1. **Setup**: `python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt`
2. **Configure**: Add your Groq API key to `.env` file
3. **Run**: `uvicorn app:app --reload`
4. **Test**: Visit `http://localhost:8000/docs` for interactive API documentation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is open source. Please check the license file for details.

## Support

For issues or questions:
- Check the logs for error details
- Review the API documentation at `/docs`
- Test with the provided sample data
- Ensure your Groq API key is valid and has sufficient credits