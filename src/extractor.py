from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from loguru import logger
import torch
import platform
import time

class SafetyNotesExtractor:
    def __init__(self, force_cpu=False):
        # Check for GPU availability
        if not force_cpu and torch.cuda.is_available():  # NVIDIA GPU
            self.device = torch.device('cuda')
            logger.info(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Properties:")
            logger.info(f"  - Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
            logger.info(f"  - CUDA Version: {torch.version.cuda}")
            
            # Add GPU memory tracking
            logger.info("Initial GPU Memory Usage:")
            logger.info(f"  - Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")
            logger.info(f"  - Reserved: {torch.cuda.memory_reserved(0)/1024**2:.2f}MB")
            
            # Force CUDA initialization with a larger operation
            warmup_tensor = torch.randn(1000, 1000).to(self.device)
            warmup_result = torch.matmul(warmup_tensor, warmup_tensor.T)
            logger.info("CUDA warmup completed")
            
        elif not force_cpu and torch.backends.mps.is_available():  # Apple Silicon
            self.device = torch.device('mps')
            logger.info(f"Using Apple Silicon GPU on {platform.processor()}")
        else:  # CPU
            self.device = torch.device('cpu')
            logger.info("Using CPU" + (" (forced)" if force_cpu else ""))
            
        # Test device with performance measurement
        start_time = time.perf_counter()
        test_tensor = torch.tensor([1.0, 2.0, 3.0]).to(self.device)
        for _ in range(1000):  # Small benchmark
            _ = test_tensor * 2
        end_time = time.perf_counter()
        logger.info(f"Test tensor on {test_tensor.device}: {test_tensor}")
        logger.info(f"Simple operation benchmark time: {(end_time - start_time)*1000:.2f}ms")
        
        # After benchmark
        if self.device.type == 'cuda':
            logger.info("GPU Memory Usage after benchmark:")
            logger.info(f"  - Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")
            logger.info(f"  - Reserved: {torch.cuda.memory_reserved(0)/1024**2:.2f}MB")
        
        self.safety_headers = {
            'en': [
                "3. Safety Notes",
                "## Safety Notes",
                "## Warning and safety instructions",
                "## Warning",
                "## Warning Batteries",
                "## Risk of electric shock",                
                "Safety Information",
                "## 2. Safety Instructions",
                "## Safety notes on charging units",
            ],
            'de': [
                "3. Sicherheitshinweise",
                "## Sicherheitshinweise",
                "Sicherheitsinformationen",
                "## Sicherheitshinweise für Ladegeräte",
            ],
            'fr': [
                "3. Consignes de sécurité",
                "## Consignes de sécurité",
                "Instructions de sécurité",
                "## Consignes de sécurité pour chargeurs",
            ]
        }
    
    def _extract_section_content(self, content: str, header: str) -> tuple[list[str], list[str]]:
        """Extract all sections matching the header pattern"""
        lines = content.split('\n')
        all_sections = []
        bullet_points = []
        current_section = []
        in_section = False
        
        for line in lines:
            # Check if this is our header
            if any(h.lower() in line.lower() for h in [header]):
                # If we were in a section, save it before starting new one
                if in_section and current_section:
                    all_sections.append('\n'.join(current_section))
                    current_section = []
                in_section = True
                current_section.append(line)
                continue
            
            # Check if we've hit the next header or end marker
            if in_section and (line.strip().startswith('#') or 
                              "Hama GmbH" in line or
                              "www.hama.com" in line or
                              line.strip().startswith('Esta fuente')):
                # Save current section before ending it
                if current_section:
                    all_sections.append('\n'.join(current_section))
                    current_section = []
                in_section = False
                continue
            
            # Track bullet points
            if in_section and line.strip().startswith('- ·'):
                bullet_points.append(line.strip())
            
            # Add content if we're in our section
            if in_section:
                current_section.append(line)
        
        # Don't forget to add the last section if we're still in one
        if in_section and current_section:
            all_sections.append('\n'.join(current_section))
        
        return all_sections, bullet_points
    
    def _find_common_bullet_count(self, contents: dict) -> int:
        """Find the number of common bullet points across languages"""
        bullet_lists = []
        for lang, (_, bullets) in contents.items():
            if bullets:  # Only consider non-empty bullet lists
                bullet_lists.append(len(bullets))
        
        if not bullet_lists:
            return 0
            
        # Find the most common bullet count
        return max(set(bullet_lists), key=bullet_lists.count)
    
    def extract_safety_notes(self, markdown_content: str, language: str = None) -> str:
        if language not in self.safety_headers:
            return "No safety information found."
        
        # Try each header pattern for the language
        for header in self.safety_headers[language]:
            sections, bullets = self._extract_section_content(markdown_content, header)
            if sections:
                for section in sections:
                    logger.info(f"Found {language} safety section: {header}")
                    return section.strip()
        
        return "No safety information found."
    
    def extract_all_languages(self, markdown_content: str) -> dict:
        """Extract safety notes for all languages"""
        contents = {}
        # Process each language
        for lang in self.safety_headers.keys():
            all_content = []
            found_sections = 0
            
            # Try each header pattern
            for header in self.safety_headers[lang]:
                sections, bullets = self._extract_section_content(markdown_content, header)
                if sections:
                    for section in sections:
                        logger.info(f"Found {lang} safety section: {header}")
                        all_content.append(section.strip())
                        found_sections += 1
            
            if found_sections > 0:
                combined_content = "\n\n".join(all_content)
                contents[lang] = (combined_content, [])
            else:
                contents[lang] = ("No safety information found.", [])
        
        return {lang: content for lang, (content, _) in contents.items()}