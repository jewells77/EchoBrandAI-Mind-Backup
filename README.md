# Creator Content Platform

A FastAPI-based platform for creators to generate and share content, leveraging LLMs and competitor analysis.

## Features
- Collects creator business DNA and competitor data
- Fetches latest competitor content (scraping/API)
- Content generation via LangChain + LangGraph (OpenAI, Gemini, etc.)
- Modular, scalable, and maintainable architecture

## Tech Stack
- FastAPI
- MongoDB
- Pydantic
- LangChain + LangGraph

## Project Structure
```
app/
  api/        # API routes
  core/       # Core settings, config, utils
  db/         # Database logic (MongoDB)
  models/     # Pydantic models
  services/   # Business logic (content, scraping)
  llm/        # LLM provider abstraction
  main.py     # FastAPI entrypoint

tests/        # Tests
requirements.txt
.env.example   # Example env vars
```

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your secrets.
3. Run the app:
   ```bash
   uvicorn app.main:app --reload
   ```
