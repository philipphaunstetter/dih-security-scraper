from loguru import logger
from src.watcher import start_watching
from src.config import settings

def main():
    logger.info("Starting Safety Notes Extractor...")
    logger.info(f"Watching directory: {settings.INPUT_DIR}")
    logger.info(f"Output directory: {settings.OUTPUT_DIR}")
    
    try:
        start_watching()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 