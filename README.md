# EcoBrandAI - Setup and Run

## Installation Steps

1. Install uv package manager:
   ```bash
   pip install uv
   ```

2. Create virtual environment:
   ```bash
   uv venv .venv --python 3.12
   ```

3. Activate the environment:
   ```bash
   source .venv/Scripts/activate
   ```

4. Install dependencies:
   ```bash
   uv sync
   ```

5. Install Playwright browser:
   ```bash
   playwright install chromium
   ```
   
   If you encounter errors related to Playwright or browser downloads, try running:
   ```bash
   playwright install
   ```

   **For Linux users:**
   If you see errors about missing browser dependencies, run:
   ```bash
   # To install Chromium dependencies only
   playwright install-deps chromium
   # Or to install all browser dependencies
   playwright install-deps
   ```

## OS-Specific Dependencies: Tesseract & PDF Processing
https://docs.unstructured.io/open-source/concepts/models

For features involving OCR (e.g., Tesseract) and PDF parsing, you must install some system dependencies.

### Windows
- Download and install Tesseract-OCR from: https://github.com/UB-Mannheim/tesseract/wiki
- After installation, add `C:\Program Files\Tesseract-OCR` to your Windows System PATH environment variable.
- To verify installation, open a new Command Prompt and run:
  ```bash
  tesseract --version
  ```

### Linux
- Install required packages with:
  ```bash
  sudo apt-get install poppler-utils tesseract-ocr libmagic-dev
  ```

### Mac (Homebrew)
- Install with:
  ```bash
  brew install poppler tesseract libmagic
  ```

## Start Command

Run the app:
```bash
uv run -m app.main
```

## Project Structure
```
app/
  api/
    deps.py
    errors.py
    exceptions.py
    router.py
    v1/
      __init__.py
      endpoints/
        chat.py
        competitors.py
        finalized_post.py
        health.py
      schemas/
        chat.py
        common.py
        competitor.py
        content.py
        creator.py
  config.py
  core/
    events.py
    logger.py
    security.py
  domain/
    agents/
      brand_analyzer.py
      competitor_analyzer.py
      final_output.py
      finalized_post_extractor.py
      image_generation_agent.py
      supervisor_agent.py
      validation_agent.py
    chains/
      brand_analysis.py
      competitor_analysis.py
      content_generation.py
    graphs/
      base_graph.py
      content_workflow.py
    llm_providers/
      base_embedding_provider.py
      base.py
      embedding_factory.py
      factory.py
      gemini_embedding_provider.py
      gemini_provider.py
      openai_provider.py
    tools/
      api_fetcher.py
      gemini_image_parser.py
      graph_visualizer.py
      image_downloader.py
      qdrant_helpers.py
      scraper.py
      text_chunker.py
      text_cleaner.py
      url_extractors.py
  infrastructure/
    blob/
      azure_media_uploader.py
    db/
      langgraph_memory.py
      mongodb.py
      repositories/
        competitor_repo.py
        content_repo.py
    http/
      http_client.py
    scraping/
      playwright_client.py
    vectorstores/
      qdrant_config.py
      qdrant_store.py
  main.py
  services/
    chat_service.py
    competitor_service.py
    embedding_service.py
    finalized_post_service.py
    provider_service.py
    qdrant_service.py

python-version
pyproject.toml
README.md
requirements.txt
uv.lock
.example.env
```
