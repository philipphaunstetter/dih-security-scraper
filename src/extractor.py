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
                "## Safety Information",
                "## 2. Safety Notes",
                "## 3. Safety Notes",
                "## 4. Safety Notes",
                "## Safety Notes",
                "## 2. Safety Instructions",
                "## 3. Safety Instructions",
                "## 4. Safety Instructions",
                "## Warning",
                "## Risk of electric shock",         
            ],
            'de': [
                "## Sicherheitshinweise",
                "## 2. Sicherheitshinweise",
                "## 3. Sicherheitshinweise",
                "## 4. Sicherheitshinweise",
                "## Warnung",
                "## Gefahr eines elektrischen Schlages",
            ],
            'fr': [
                "## Consignes de sécurité",
                "## 2. Consignes de sécurité",
                "## 3. Consignes de sécurité",
                "## 4. Consignes de sécurité",
                "## Risque d’électrocution",
                "## Avertissement",
            ],
            'es': [
                "## Indicaciones de seguridad",
                "## 2. Indicaciones de seguridad",
                "## 3. Indicaciones de seguridad",
                "## 4. Indicaciones de seguridad",
                "## Peligro de sufrir una descarga eléctrica",
                "## Aviso",
            ],
            'ru': [
                "## Указания по технике безопасности",
                "## 2. Указания по технике безопасности",
                "## 3. Указания по технике безопасности",
                "## 4. Указания по технике безопасности",
                "## Предупреждение",
                "## Опасность поражения электрическим током",
            ],
            'it': [
                "## Istruzioni di sicurezza",
                "## 2. Istruzioni di sicurezza",
                "## 3. Istruzioni di sicurezza",
                "## 4. Istruzioni di sicurezza",
                "## Pericolo di scarica elettrica",
                "## Attenzione",
            ]
        }
        
        # Add ignore phrases for each language
        self.ignore_phrases = {
            'en': [
                "This symbol indicates a risk of electric shock from touching product parts carrying hazardous voltage.",
                "This symbol is used to indicate safety instructions or to draw your attention to specific hazards and risks",
                "This symbol is used to indicate additional information or important notes.",
            ],
            'de': [
                "Dieses Symbol weist auf eine Berührungsgefahr mit nicht isolierten Teilen des Produktes hin, die möglicherweise eine gefährliche Spannung von solcher Höhe führen, dass die Gefahr eines elektrischen Schlags besteht.",
                "Wird verwendet, um Sicherheitshinweise zu kennzeichnen oder um Aufmerksamkeit auf besondere Gefahren und Risiken zu lenken.",
                "Wird verwendet, um zusätzlich Informationen oder wichtige Hinweise zu kennzeichnen.",
            ],
            'fr': [
                "Ce symbole indique un risque de contact avec des parties non isolées du produit susceptibles de conduire un courant électrique capable de provoquer une électrocution.",
                "Ce symbole est utilisé pour indiquer des consignes de sécurité ou pour attirer votre attention sur des dangers et risques particuliers.",
                "Ce symbole est utilisé pour indiquer des informations supplémentaires ou des remarques importantes.",
            ],
            'es': [
                "Este símbolo hace referencia al peligro de contacto con partes no aisladas del producto que pueden conducir una tensión peligrosa de una intensidad tal que puede provocar una descarga eléctrica.",
                "Se utiliza para indicar indicaciones de seguridad o para llamar la atención sobre peligros y riesgos especiales.",
                "Se utiliza para indicar información adicional o indicaciones importantes.",
            ],
            'ru': [
                "Данный символ указывает на опасность соприкосновения с неизолированными частями изделия под напряжением, которое может привести к поражению электрическим током.",
                "Используется для обозначения указаний по технике безопасности или для акцентирования внимания на особых опасностях и рисках.",
                "Используется для дополнительного обозначения информации или важных указаний.",
            ],
            'it': [
                "Questo simbolo indica la presenza di pericoli dovuti al contatto con parti del prodotto sotto tensione, di entità tale da comportare il rischio di scarica elettrica.",
                "Contraddistingue le istruzioni di sicurezza o richiama l’attenzione su particolari rischi e pericoli.",
                "Contraddistingue informazioni supplementari o indicazioni importanti.",
            ]
        }
    
    def _extract_section_content(self, content: str, header: str) -> tuple[list[str], list[str]]:
        """Extract all sections matching the header pattern"""
        lines = content.split('\n')
        all_sections = []
        bullet_points = []
        current_section = []
        in_section = False
        skip_current_section = False
        
        for line in lines:
            # Check if this is our header
            if any(h.lower() in line.lower() for h in [header]):
                # If we were in a section, save it before starting new one
                if in_section and current_section and not skip_current_section:
                    all_sections.append('\n'.join(current_section))
                current_section = []
                in_section = True
                skip_current_section = False
                current_section.append(line)
                logger.debug(f"Found header: {line.strip()}")
                continue
            
            # Check if we should ignore this section based on the next line after header
            if in_section and len(current_section) == 1:  # We just added the header
                detected_lang = self._detect_language(header)
                for ignore_phrase in self.ignore_phrases.get(detected_lang, []):
                    if ignore_phrase.lower() in line.lower():
                        logger.info(f"Ignoring section due to phrase: '{ignore_phrase}'")
                        skip_current_section = True
                        in_section = False
                        current_section = []
                        break
                
                if not skip_current_section:
                    logger.debug(f"Section accepted, continuing extraction")
            
            # Only process the rest if we're not skipping
            if not skip_current_section:
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
        if in_section and current_section and not skip_current_section:
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
                    # Clean the content before returning
                    return self.clean_content(section)
        
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
                        # Clean each section before adding
                        all_content.append(self.clean_content(section))
                        found_sections += 1
            
            if found_sections > 0:
                combined_content = "\n\n".join(all_content)
                contents[lang] = (combined_content, [])
            else:
                contents[lang] = ("No safety information found.", [])
        
        return {lang: content for lang, (content, _) in contents.items()}
    
    def clean_content(self, content: str) -> str:
        # First remove image tags
        lines = content.split('\n')
        cleaned_lines = [line for line in lines if "<!-- image -->" not in line]
        content = '\n'.join(cleaned_lines)
        
        # Remove bullet points "· "
        content = content.replace("· ", "")
        
        # Then remove sections with ignore phrases
        lines = content.split('\n')
        cleaned_content = []
        skip_next_header = False
        last_header = None
        current_language = None
        
        for i, line in enumerate(lines):
            current_line = line.strip()
            
            # Track headers and detect language
            if current_line.startswith('##'):
                if skip_next_header:
                    skip_next_header = False
                    logger.debug(f"Skipping header: {line}")
                    continue
                last_header = i
                # Detect language from header
                for lang, headers in self.safety_headers.items():
                    if any(h.lower() in current_line.lower() for h in headers):
                        current_language = lang
                        logger.debug(f"Language set to {lang} for header: {current_line}")
                        break
            
            # Check for ignore phrases using current language
            if current_language:
                for phrase in self.ignore_phrases.get(current_language, []):
                    if phrase.lower() in current_line.lower():
                        logger.info(f"Found ignore phrase '{phrase}' in language {current_language}")
                        if last_header is not None:
                            logger.info(f"Removing content from header at line {last_header}")
                            cleaned_content = cleaned_content[:last_header]
                        skip_next_header = True
                        break
                else:
                    cleaned_content.append(line)
            else:
                cleaned_content.append(line)
        
        # Process the cleaned content with proper spacing
        final_lines = []
        prev_was_header = False
        prev_was_empty = False
        
        for line in cleaned_content:
            current_line = line.strip()
            is_header = current_line.startswith('##')
            is_empty = not current_line
            
            if is_header:
                # Add single empty line before header unless it's the first line
                if final_lines and not prev_was_empty:
                    final_lines.append('')
                final_lines.append(line)
                prev_was_header = True
            elif is_empty:
                # Only add empty line if previous line wasn't empty or header
                if not prev_was_empty and not prev_was_header:
                    final_lines.append('')
                prev_was_empty = True
            else:
                final_lines.append(line)
                prev_was_empty = False
            
            if not is_empty:
                prev_was_header = is_header
        
        return '\n'.join(final_lines).strip()
    
    def extract_language_blocks(self, content: str) -> dict:
        # ... existing code ...
        for match in matches:
            language = match.group(1)
            block_content = match.group(2)
            # Clean the content before adding to the dictionary
            blocks[language] = self.clean_content(block_content)
        return blocks
    
    def _detect_language(self, header: str) -> str:
        """Detect language based on the header"""
        for lang, headers in self.safety_headers.items():
            if any(h.lower() in header.lower() for h in headers):
                logger.debug(f"Detected language: {lang} for header: {header}")
                return lang
        logger.debug(f"No language detected for header: {header}, defaulting to 'en'")
        return 'en'  # default to English if not found