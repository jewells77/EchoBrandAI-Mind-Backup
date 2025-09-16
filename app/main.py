from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.config import settings
from app.core.events import startup_event_handler, shutdown_event_handler


# Define lifespan context
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup_event_handler()
    yield
    # Shutdown
    await shutdown_event_handler()


# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for EcoBrandAI - AI-powered content creation for eco-conscious brands",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    debug=settings.DEBUG,
    lifespan=lifespan,
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api")


# Health check endpoint at root
@app.get("/")
async def root():
    return {"status": "ok", "message": f"Welcome to {settings.PROJECT_NAME} API"}


if __name__ == "__main__":
    import uvicorn

    # ⚠️ Note: When using Playwright for scraping, set reload=False.
    # Reason: Uvicorn's reload=True (auto-reload on code changes) interferes with
    # Playwright's subprocess management and can raise NotImplementedError.
    # Tip: Use reload=True only during normal development (without Playwright).
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
