from pathlib import Path
from loguru import logger
from docling.document_converter import DocumentConverter
import shutil
from typing import Optional, Dict
from .config import Settings
from .extractor import SafetyNotesExtractor

class PDFProcessor:
    def __init__(self, config: Settings, extractor: SafetyNotesExtractor = None):
        self.config = config
        self.converter = DocumentConverter()
        self.extractor = extractor or SafetyNotesExtractor()
        self.image_only_file = self.config.OUTPUT_DIR / "no_real_text.md"
    
    def _is_mostly_images(self, content: str, threshold: float = 0.8) -> bool:
        """Check if content is mostly image tags"""
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return True
            
        image_lines = [line for line in lines if line == "<!-- image -->"]
        image_ratio = len(image_lines) / len(lines)
        
        logger.debug(f"Image ratio: {image_ratio:.2f} ({len(image_lines)}/{len(lines)} lines)")
        return image_ratio >= threshold
    
    def _append_to_image_only_list(self, filename: str):
        """Append filename to the list of image-only files"""
        with self.image_only_file.open('a', encoding='utf-8') as f:
            f.write(f"{filename}\n")
        logger.warning(f"Added {filename} to image-only list")
    
    def _get_article_folder(self, pdf_path: Path) -> Path:
        """Create or get article folder based on first 8 characters of filename"""
        article_number = pdf_path.stem[:8]  # Get first 8 chars of filename
        article_folder = self.config.OUTPUT_DIR / article_number
        
        # Create folder if it doesn't exist
        article_folder.mkdir(exist_ok=True)
        
        logger.info(f"Using article folder: {article_folder}")
        return article_folder
    
    def _save_safety_notes(self, safety_notes: Dict[str, str], base_path: Path):
        """Save safety notes to separate files per language"""
        article_folder = self._get_article_folder(base_path)
        for lang, content in safety_notes.items():
            if content and content != "No safety information found.":
                safety_output_path = article_folder / f"{base_path.stem}_safety_{lang}.md"
                logger.info(f"Saving {lang} safety notes to {safety_output_path}")
                safety_output_path.write_text(content, encoding='utf-8')
    
    def process_pdf(self, pdf_path: Path) -> Optional[Path]:
        """Process a PDF file and extract its content"""
        try:
            # Convert PDF to docling document
            result = self.converter.convert(str(pdf_path))
            
            # Export to markdown
            markdown_content = result.document.export_to_markdown()
            
            # Check if content is mostly images
            if self._is_mostly_images(markdown_content):
                self._append_to_image_only_list(pdf_path.name)
            
            # Get article folder
            article_folder = self._get_article_folder(pdf_path)
            
            # Create output file path for full content
            full_output_path = article_folder / f"{pdf_path.stem}_full.md"
            
            # Save full markdown content
            full_output_path.write_text(markdown_content, encoding='utf-8')
            
            # Extract safety notes for each language
            logger.info("Extracting and normalizing safety notes for all languages")
            safety_notes = self.extractor.extract_all_languages(markdown_content)
            
            # Save safety notes per language
            self._save_safety_notes(safety_notes, pdf_path)
            
            # Move processed PDF
            processed_folder = self.config.PROCESSED_DIR / pdf_path.stem[:8]
            processed_folder.mkdir(exist_ok=True)
            shutil.move(
                pdf_path,
                processed_folder / pdf_path.name
            )
            
            return full_output_path
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            # Move failed PDF
            failed_folder = self.config.FAILED_DIR / pdf_path.stem[:8]
            failed_folder.mkdir(exist_ok=True)
            shutil.move(
                pdf_path,
                failed_folder / pdf_path.name
            )
            return None 