"""
Content processor for extracting and preprocessing text from various sources.

This module handles text extraction, encoding detection, and preprocessing
for different file types before embedding generation.
"""

import logging
from typing import Optional, Tuple
import chardet
import magic

logger = logging.getLogger(__name__)


class ContentProcessor:
    """
    Process and extract text content from objects.
    
    Handles encoding detection, text extraction, and preprocessing
    for semantic indexing.
    """
    
    def __init__(
        self,
        max_file_size_mb: int = 100,
        encoding_detection: bool = True,
        fallback_encoding: str = "utf-8",
        supported_extensions: Optional[list] = None
    ):
        """
        Initialize content processor.
        
        Args:
            max_file_size_mb: Maximum file size to process (MB)
            encoding_detection: Whether to auto-detect encoding
            fallback_encoding: Encoding to use if detection fails
            supported_extensions: List of supported file extensions
        """
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.encoding_detection = encoding_detection
        self.fallback_encoding = fallback_encoding
        
        if supported_extensions is None:
            self.supported_extensions = {
                'txt', 'md', 'py', 'js', 'java', 'cpp', 'c', 'h',
                'json', 'yaml', 'yml', 'xml', 'csv', 'log',
                'sh', 'bash', 'rst', 'tex'
            }
        else:
            self.supported_extensions = set(supported_extensions)
        
        logger.info(f"Initialized ContentProcessor (max_size={max_file_size_mb}MB)")
    
    def detect_encoding(self, data: bytes) -> str:
        """
        Detect text encoding using chardet.
        
        Args:
            data: Raw bytes
            
        Returns:
            Detected encoding name
        """
        if not self.encoding_detection:
            return self.fallback_encoding
        
        try:
            # Sample first 10KB for performance
            sample = data[:10240]
            result = chardet.detect(sample)
            
            encoding = result.get('encoding', self.fallback_encoding)
            confidence = result.get('confidence', 0.0)
            
            logger.debug(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
            
            # Use fallback if confidence is too low
            if confidence < 0.7:
                logger.warning(f"Low confidence ({confidence:.2f}), using fallback: {self.fallback_encoding}")
                return self.fallback_encoding
            
            return encoding or self.fallback_encoding
            
        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}, using fallback")
            return self.fallback_encoding
    
    def detect_content_type(self, data: bytes, filename: Optional[str] = None) -> str:
        """
        Detect content type using python-magic.
        
        Args:
            data: Raw bytes
            filename: Optional filename for extension-based detection
            
        Returns:
            Content type string
        """
        try:
            # Try magic-based detection
            mime = magic.from_buffer(data, mime=True)
            logger.debug(f"Detected MIME type: {mime}")
            return mime
        except Exception as e:
            logger.warning(f"MIME detection failed: {e}")
            
            # Fall back to extension-based detection
            if filename:
                ext = filename.rsplit('.', 1)[-1].lower()
                return f"text/{ext}" if ext in self.supported_extensions else "application/octet-stream"
            
            return "text/plain"
    
    def is_text_file(self, content_type: str) -> bool:
        """
        Check if content type is text-based.
        
        Args:
            content_type: MIME type or content type string
            
        Returns:
            True if text-based, False otherwise
        """
        text_types = {'text/', 'application/json', 'application/xml', 'application/yaml'}
        return any(content_type.startswith(t) for t in text_types)
    
    def is_supported(self, object_name: str) -> bool:
        """
        Check if object is supported for text extraction.
        
        Args:
            object_name: Name of the object
            
        Returns:
            True if supported, False otherwise
        """
        # Check extension
        if '.' in object_name:
            ext = object_name.rsplit('.', 1)[-1].lower()
            # Only treat as extension if it's alphanumeric and reasonable length
            # This avoids treating CephFS internal names like "10000000000.00000000" as having extensions
            if ext.isalpha() and len(ext) <= 10:
                return ext in self.supported_extensions
            # If it looks like a numeric suffix or other pattern, treat as no extension
            # and check content type during extraction
        
        # No valid extension, assume supported (will check content type later)
        return True
    
    def extract_text(
        self,
        data: bytes,
        object_name: Optional[str] = None,
        max_size: Optional[int] = None
    ) -> Tuple[str, str]:
        """
        Extract text content from raw bytes.
        
        Args:
            data: Raw bytes
            object_name: Optional object name for type detection
            max_size: Maximum size to process (overrides default)
            
        Returns:
            Tuple of (extracted_text, encoding_used)
            
        Raises:
            ValueError: If file is too large or unsupported
        """
        # Check size
        max_bytes = max_size or self.max_file_size_bytes
        if len(data) > max_bytes:
            raise ValueError(
                f"File too large: {len(data)} bytes (max: {max_bytes})"
            )
        
        # Detect content type
        content_type = self.detect_content_type(data, object_name)
        
        if not self.is_text_file(content_type):
            raise ValueError(f"Unsupported content type: {content_type}")
        
        # Detect encoding
        encoding = self.detect_encoding(data)
        
        # Decode text
        try:
            text = data.decode(encoding)
            logger.debug(f"Successfully decoded {len(data)} bytes as {encoding}")
            return text, encoding
            
        except UnicodeDecodeError as e:
            logger.warning(f"Failed to decode with {encoding}: {e}")
            
            # Try fallback encoding
            try:
                text = data.decode(self.fallback_encoding, errors='ignore')
                logger.info(f"Decoded with fallback encoding: {self.fallback_encoding}")
                return text, self.fallback_encoding
                
            except Exception as e2:
                logger.error(f"All decoding attempts failed: {e2}")
                raise ValueError(f"Cannot decode file: {e2}")
    
    def preprocess_text(self, text: str, max_length: Optional[int] = None) -> str:
        """
        Preprocess text for embedding generation.
        
        Args:
            text: Raw text
            max_length: Maximum length (characters)
            
        Returns:
            Preprocessed text
        """
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if needed
        if max_length and len(text) > max_length:
            text = text[:max_length]
            logger.debug(f"Truncated text to {max_length} characters")
        
        return text
    
    def create_content_preview(self, text: str, length: int = 500) -> str:
        """
        Create a preview of the content.
        
        Args:
            text: Full text
            length: Preview length in characters
            
        Returns:
            Preview text
        """
        if len(text) <= length:
            return text
        
        # Try to cut at word boundary
        preview = text[:length]
        last_space = preview.rfind(' ')
        
        if last_space > length * 0.8:  # If we have a space in the last 20%
            preview = preview[:last_space]
        
        return preview + "..."
    
    def chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> list[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk (characters)
            overlap: Overlap between chunks (characters)
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end in the last 20% of chunk
                search_start = int(chunk_size * 0.8)
                for delimiter in ['. ', '.\n', '! ', '?\n', '? ']:
                    pos = chunk.rfind(delimiter, search_start)
                    if pos > 0:
                        chunk = chunk[:pos + 1]
                        break
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def get_stats(self, text: str) -> dict:
        """
        Get statistics about text content.
        
        Args:
            text: Text content
            
        Returns:
            Dictionary with stats (characters, words, lines)
        """
        return {
            'characters': len(text),
            'words': len(text.split()),
            'lines': text.count('\n') + 1,
            'bytes': len(text.encode('utf-8'))
        }
