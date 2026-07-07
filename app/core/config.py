from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    """
    Application configuration settings.
    Loads from environment variables or .env file.
    """
    PROJECT_NAME: str = "AI Document Classification API"
    API_V1_STR: str = "/api/v1"
    
    # Model configuration
    MODEL_NAME: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    USE_GPU_IF_AVAILABLE: bool = True
    
    # File handling
    MAX_FILE_SIZE_MB: int = 10
    UPLOAD_DIR: str = "uploads"
    ALLOWED_EXTENSIONS: List[str] = ["pdf", "jpg", "jpeg", "png"]
    
    # Pydantic v2 specific configuration for env files
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
