from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
from loguru import logger
import time
from .processor import PDFProcessor
from .config import settings

class PDFHandler(FileSystemEventHandler):
    def __init__(self, extractor):
        self.processor = PDFProcessor(settings, extractor)
        self._process_existing_files()
    
    def _process_existing_files(self):
        """Process any existing PDF files in the input directory"""
        for pdf_file in settings.INPUT_DIR.glob("*.pdf"):
            logger.info(f"Processing existing PDF: {pdf_file}")
            self.processor.process_pdf(pdf_file)
    
    def on_created(self, event):
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        if file_path.suffix.lower() == '.pdf':
            logger.info(f"New PDF detected: {file_path}")
            self.processor.process_pdf(file_path)

def start_watching(extractor):
    event_handler = PDFHandler(extractor)
    observer = Observer()
    observer.schedule(event_handler, str(settings.INPUT_DIR), recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join() 