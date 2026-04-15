"""Run the FastAPI app via Uvicorn."""

import uvicorn
from config.settings import Settings

settings = Settings()


if __name__ == "__main__":
    uvicorn.run("app.api:app", host=settings.API_HOST, port=settings.API_PORT, log_level="info")
