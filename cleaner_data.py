import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer


import unicodedata

nlp = spacy.load('es_core_news_md')

class AdvancedTextCleaner:
    def __init__(self, language='spanish'):
        self.language = language
        self.stop_words = set(stopwords.words(language))
        # Cargar modelo de transformers para análisis semántico
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        self.model = AutoModel.from_pretrained("BAAI/bge-m3")
        
    def remove_punctuation_patterns(self, text):
        """Elimina patrones de puntuación repetitivos como '....'"""
        # Eliminar múltiples puntos seguidos
        text = re.sub(r'\.{2,}', ' ', text)
        # Eliminar patrones de puntuación que forman líneas
        text = re.sub(r'[.=_\-]{3,}', ' ', text)
        return text
    
    def remove_page_numbers(self, text):
        """Elimina números de página y referencias"""
        # Eliminar números de página típicos
        text = re.sub(r'\b\d+\b(?=\s*$)', '', text)
        # Eliminar referencias de formato [número]
        text = re.sub(r'\[\d+\]', '', text)
        return text
    
    def remove_section_numbering(self, text):
        """Elimina numeración de secciones como '3.5.2.' o '4.1.1.'"""
        text = re.sub(r'\b\d+(\.\d+)*\.?\s+', '', text)
        return text
    
    def normalize_whitespace(self, text):
        """Normaliza espacios en blanco"""
        # Reemplazar múltiples espacios en blanco por uno solo
        text = re.sub(r'\s+', ' ', text)
        # Eliminar espacios al inicio y final
        return text.strip()
    
    def remove_headers_footers(self, text):
        """Elimina encabezados y pies de página típicos de documentos académicos"""
        # Eliminar líneas con "Figura X:"
        text = re.sub(r'Figura \d+(\.\d+)*:.*?\n', '', text)
        # Eliminar líneas que son solo números (páginas)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        return text
    
    def remove_equation_numbers(self, text):
        """Elimina números de ecuaciones como (3.1), (3.2)"""
        text = re.sub(r'\(\d+\.\d+\)', '', text)
        return text
    
    def normalize_accents_and_special_chars(self, text):
        """Normaliza acentos y caracteres especiales"""
        # Crear mapeo de caracteres con acentos a sin acentos
        accents_map = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
            'ñ': 'n', 'Ñ': 'N',
            'ü': 'u', 'Ü': 'U'
        }
        
        # Reemplazar caracteres especiales según el mapeo
        for accentuated, plain in accents_map.items():
            text = text.replace(accentuated, plain)
            
        # Usar unicodedata para normalizar caracteres que no estén en el mapeo
        text = ''.join(c for c in unicodedata.normalize('NFD', text)
                      if unicodedata.category(c) != 'Mn')
        
        return text
        
    def remove_formula_artifacts(self, text):
        """Elimina artefactos comunes en fórmulas matemáticas"""
        # Limpiar artefactos comunes
        text = re.sub(r'[\{\}\\]', '', text)
        return text
        
    def clean_text(self, text):
        """Proceso completo de limpieza para texto académico"""
        # Aplicar todas las técnicas de limpieza
        text = self.remove_punctuation_patterns(text)
        text = self.remove_page_numbers(text)
        text = self.remove_section_numbering(text)
        text = self.remove_headers_footers(text)
        text = self.remove_equation_numbers(text)
        text = self.remove_formula_artifacts(text)

        # Análisis más profundo usando spaCy
        doc = nlp(text)
        
        # Filtrar tokens significativos
        filtered_tokens = []
        for token in doc:
            # Mantener solo tokens que no sean espacios en blanco, puntuaciones o stopwords
            if (not token.is_punct and not token.is_space 
                and not token.like_num and len(token.text) > 1):
                filtered_tokens.append(token.text)
                
        # Reconstruir texto con tokens importantes
        clean_text = ' '.join(filtered_tokens)
        clean_text = self.normalize_whitespace(clean_text)
        
        # Normalizar acentos y caracteres especiales
        clean_text = self.normalize_accents_and_special_chars(clean_text)
        
        
    
        return clean_text

# Demostración del uso
text_cleaner = AdvancedTextCleaner()