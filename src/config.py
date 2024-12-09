from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Folder paths
    INPUT_DIR: Path = Path("input")
    OUTPUT_DIR: Path = Path("output")
    PROCESSED_DIR: Path = Path("processed")
    FAILED_DIR: Path = Path("failed")
    MODELS_DIR: Path = Path("models")
    
    # Extraction settings
    SIMILARITY_THRESHOLD: float = 0.4  # Threshold for semantic similarity matching
    
    # Language detection confidence threshold
    LANG_CONFIDENCE_THRESHOLD: float = 0.8
    
    # Create directories if they don't exist
    def create_directories(self):
        for dir_path in [self.INPUT_DIR, self.OUTPUT_DIR, 
                        self.PROCESSED_DIR, self.FAILED_DIR,
                        self.MODELS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = ".env"

settings = Settings()
settings.create_directories() 