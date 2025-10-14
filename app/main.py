import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse
from app.api.router import api_router
from app.config import settings
from app.core.events import startup_event_handler, shutdown_event_handler
from app.api.exceptions import APIError


# Define lifespan context
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup_event_handler(app)
    yield
    # Shutdown
    await shutdown_event_handler(app)


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
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api")


# Custom exception handler for APIError
@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message": exc.message,
            "public_message": exc.public_message,
        },
    )


# Health check endpoint at root
@app.get("/")
async def root():
    return {"status": "ok", "message": f"Welcome to {settings.PROJECT_NAME} API"}


if __name__ == "__main__":

    # ⚠️ Note: When using Playwright for scraping, set reload=False.
    # Reason: Uvicorn's reload=True (auto-reload on code changes) interferes with
    # Playwright's subprocess management and can raise NotImplementedError.
    # Tip: Use reload=True only during normal development (without Playwright).
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
