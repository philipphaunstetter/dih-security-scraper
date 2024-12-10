from loguru import logger
from src.watcher import start_watching
from src.config import settings
from src.extractor import SafetyNotesExtractor
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force-cpu', action='store_true', help='Force CPU usage instead of GPU')
    args = parser.parse_args()
    
    logger.info("Starting Safety Notes Extractor...")
    logger.info(f"Watching directory: {settings.INPUT_DIR}")
    logger.info(f"Output directory: {settings.OUTPUT_DIR}")
    
    try:
        extractor = SafetyNotesExtractor(force_cpu=args.force_cpu)
        start_watching(extractor=extractor)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()