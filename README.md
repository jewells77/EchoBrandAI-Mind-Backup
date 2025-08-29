# EcoBrandAI - Content Generation Platform

A FastAPI-based platform for creators to generate and share content, leveraging LLMs and competitor analysis with a multi-agent architecture using LangChain and LangGraph.

## Features
- 🧠 Multi-agent architecture with specialized roles:
  - Brand DNA Analyzer Agent
  - Competitor Intelligence Agent
  - Content Strategist Agent
  - Content Generator Agent
  - Content Refiner Agent
- 🔄 Parallel processing with LangGraph
- 🕸️ Web scraping for competitor analysis (Playwright)
- 🤖 Advanced LLM-powered content creation (OpenAI)
- 📈 Structured content strategy with customizable outputs
- 🧩 Modular, scalable, and maintainable architecture

## Tech Stack
- FastAPI
- MongoDB
- Pydantic
- LangChain
- LangGraph
- Playwright (scraping)
- OpenAI (LLM)

## Multi-Agent Architecture

The system uses 5 specialized agents working together:

1. **Brand DNA Analyzer Agent**  
   - Input: Brand details + competitor list
   - Task: Extract brand tone, audience, positioning
   - Output: Brand persona profile (JSON)

2. **Competitor Intelligence Agent**  
   - Input: Competitor URLs
   - Task: Scrape and analyze competitor content
   - Output: Insights and content gap opportunities

3. **Content Strategist Agent**  
   - Input: Brand profile + competitor insights + content request
   - Task: Develop content strategy
   - Output: Strategy with titles, formats, angles

4. **Content Generator Agent**  
   - Input: Theme and format
   - Task: Generate draft content
   - Output: Content draft

5. **Content Refiner Agent**  
   - Input: Draft + guidelines
   - Task: Polish for brand consistency
   - Output: Final content

## Workflow Diagram

```
[User Input: Brand Details + Competitors + Content Request]
        │
        ▼
 ┌──────────────────┐
 │ Brand DNA Analyzer│
 └──────────────────┘
        │
        ▼
 ┌─────────────────────────┐
 │ Competitor Intelligence │───┐
 └─────────────────────────┘   │ (Parallel)
        │                       │
        └──────┬────────────────┘
               ▼
 ┌─────────────────────────┐
 │ Content Strategist Agent│
 └─────────────────────────┘
        │
        ▼
 ┌─────────────────────────┐
 │ Content Generator Agent │
 └─────────────────────────┘
        │
        ▼
 ┌─────────────────────────┐
 │ Content Refiner Agent   │
 └─────────────────────────┘
        │
        ▼
   [Final Content]
```

## Project Structure
```
app/
  api/                # API routes and schemas
    v1/
      endpoints/      # API endpoints
      schemas/        # Pydantic models for API
  core/               # Core settings, logging, security
  domain/             # Domain logic
    agents/           # Multi-agent implementation
    chains/           # LangChain chains
    graphs/           # LangGraph workflows
    llm_providers/    # LLM provider abstraction
    tools/            # Agent tools
  infrastructure/     # External services
    db/               # Database (MongoDB)
    scraping/         # Web scraping (Playwright)
    vectorstores/     # Vector databases
  services/           # Business logic
  main.py             # FastAPI entrypoint

tests/                # Tests
requirements.txt
.env.example         # Example env vars
```

## Setup and Running Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install Playwright browsers:
   ```bash
   playwright install chromium
   ```

3. Copy `env.example` to `.env` and fill in your secrets:
   ```
   OPENAI_API_KEY=your_openai_key
   OPENAI_MODEL_NAME=gpt-4-turbo  # or your preferred model
   ```

4. Make sure MongoDB is running locally or update the MongoDB connection string in your `.env` file:
   ```
   MONGODB_URI=mongodb://localhost:27017  # default
   MONGODB_DB_NAME=ecobrandai  # default
   ```

5. Run the app using one of these methods:

   **Method 1 - Direct Python execution:**
   ```bash
   python -m app.main
   ```

   **Method 2 - Using Uvicorn (recommended for development):**
   ```bash
   uvicorn app.main:app --reload
   ```

6. Access the API documentation:
   ```
   http://localhost:8000/docs  # Swagger UI
   http://localhost:8000/redoc  # ReDoc UI
   ```

7. Check the application is running:
   ```
   http://localhost:8000/  # Should display a welcome message
   http://localhost:8000/api/health  # Should display health status
   ```

## API Usage

Generate content with the multi-agent system:

```python
import requests

url = "http://localhost:8000/api/v1/content/generate"

# Basic request with required parameters only
minimal_payload = {
    "brand_details": {
        "name": "EcoGreen Solutions",
        "description": "Sustainable home products that reduce waste",
        "industry": "Home Goods"
    },
    "content_request": "Create a blog post about reducing plastic use in kitchen"
}

# Full request with all parameters (competitors and guidelines are optional)
full_payload = {
    "brand_details": {
        "name": "EcoGreen Solutions",
        "description": "Sustainable home products that reduce waste",
        "values": ["sustainability", "innovation", "quality"],  # Optional
        "industry": "Home Goods",
        "mission_statement": "Helping homes reduce waste with eco-friendly products"  # Optional
    },
    "competitors": [  # Optional - can be omitted
        "https://example.com/competitor1",
        "https://example.com/competitor2"
    ],
    "content_request": "Create a blog post about reducing plastic use in kitchen",
    "guidelines": {  # Optional - can be omitted
        "tone": "informative but friendly",
        "target_audience": "environmentally-conscious homeowners"
    }
}

response = requests.post(url, json=full_payload)
print(response.json())
```

## LangGraph Implementation

For parallel processing and advanced agent orchestration, the `/api/v1/content/langgraph` endpoint uses LangGraph to run the Brand DNA Analyzer and Competitor Intelligence agents in parallel before merging their outputs for the rest of the workflow.
