from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from loguru import logger

class SafetyNotesExtractor:
    def __init__(self):
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
                "Safety Instructions",
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