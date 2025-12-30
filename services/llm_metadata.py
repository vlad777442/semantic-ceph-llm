"""
LLM-based metadata generation service.

This module uses LLM (Ollama/OpenAI) to automatically generate rich metadata
for uploaded files: summary, keywords, tags, and content classification.
"""

import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from core.llm_provider import BaseLLMProvider, create_llm_provider

logger = logging.getLogger(__name__)


@dataclass
class GeneratedMetadata:
    """Container for LLM-generated metadata."""
    summary: str
    keywords: List[str]
    tags: List[str]
    content_type_hint: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "keywords": self.keywords,
            "tags": self.tags,
            "content_type_hint": self.content_type_hint
        }


class LLMMetadataGenerator:
    """
    Service for generating metadata using LLM.
    
    Analyzes file content and generates:
    - Summary: A concise description of the content
    - Keywords: Relevant search terms (5-10)
    - Tags: Category labels for classification
    """
    
    SYSTEM_PROMPT = """You are a document analysis assistant. Your job is to analyze file content and generate metadata that will help with semantic search and organization.

Be concise but informative. Extract the key concepts and themes from the content."""

    METADATA_PROMPT_TEMPLATE = """Analyze the following file content and generate metadata for semantic search.

File name: {filename}
{user_description_section}

Content (first {max_chars} characters):
---
{content}
---

Generate a JSON response with:
1. "summary": A 1-2 sentence description of what this file contains and its purpose
2. "keywords": An array of 5-10 relevant search terms/concepts from the content
3. "tags": An array of 2-5 category labels (e.g., "documentation", "code", "config", "data", "report")
4. "content_type_hint": A brief classification (e.g., "python code", "markdown documentation", "json config")

Respond ONLY with valid JSON, no other text."""

    def __init__(
        self,
        llm_provider: Optional[BaseLLMProvider] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        max_content_chars: int = 4000
    ):
        """
        Initialize the metadata generator.
        
        Args:
            llm_provider: Pre-configured LLM provider instance
            llm_config: Config dict to create provider (if provider not given)
            max_content_chars: Maximum characters of content to send to LLM
        """
        if llm_provider:
            self.llm = llm_provider
        elif llm_config:
            self.llm = create_llm_provider(llm_config)
        else:
            raise ValueError("Either llm_provider or llm_config must be provided")
        
        self.max_content_chars = max_content_chars
        logger.info(f"Initialized LLMMetadataGenerator with model: {self.llm.model}")
    
    def generate(
        self,
        content: str,
        filename: str,
        user_description: Optional[str] = None
    ) -> GeneratedMetadata:
        """
        Generate metadata for file content using LLM.
        
        Args:
            content: The text content of the file
            filename: Name of the file (helps with context)
            user_description: Optional user-provided description hint
            
        Returns:
            GeneratedMetadata with summary, keywords, tags
        """
        try:
            # Truncate content if too long
            truncated_content = content[:self.max_content_chars]
            if len(content) > self.max_content_chars:
                truncated_content += "\n... [content truncated]"
            
            # Build user description section
            user_desc_section = ""
            if user_description:
                user_desc_section = f"User-provided description: {user_description}\n"
            
            # Build prompt
            prompt = self.METADATA_PROMPT_TEMPLATE.format(
                filename=filename,
                user_description_section=user_desc_section,
                max_chars=self.max_content_chars,
                content=truncated_content
            )
            
            # Call LLM
            logger.debug(f"Generating metadata for: {filename}")
            response = self.llm.complete(prompt, system=self.SYSTEM_PROMPT)
            
            # Parse response
            metadata = self._parse_response(response, filename, user_description)
            
            logger.info(f"Generated metadata for {filename}: {len(metadata.keywords)} keywords, {len(metadata.tags)} tags")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to generate metadata for {filename}: {e}")
            # Return fallback metadata
            return self._fallback_metadata(filename, user_description)
    
    def _parse_response(
        self,
        response: str,
        filename: str,
        user_description: Optional[str]
    ) -> GeneratedMetadata:
        """Parse LLM response into GeneratedMetadata."""
        try:
            # Try to parse as JSON
            data = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse LLM response as JSON")
                    return self._fallback_metadata(filename, user_description)
            else:
                logger.warning(f"No JSON found in LLM response")
                return self._fallback_metadata(filename, user_description)
        
        # Extract fields with defaults
        summary = data.get("summary", "")
        keywords = data.get("keywords", [])
        tags = data.get("tags", [])
        content_type_hint = data.get("content_type_hint", "")
        
        # Ensure keywords and tags are lists
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",")]
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]
        
        return GeneratedMetadata(
            summary=summary,
            keywords=keywords[:10],  # Limit to 10 keywords
            tags=tags[:5],  # Limit to 5 tags
            content_type_hint=content_type_hint
        )
    
    def _fallback_metadata(
        self,
        filename: str,
        user_description: Optional[str]
    ) -> GeneratedMetadata:
        """Generate fallback metadata when LLM fails."""
        # Extract basic info from filename
        import os
        name, ext = os.path.splitext(filename)
        
        summary = user_description or f"File: {filename}"
        keywords = [name.replace("_", " ").replace("-", " ")]
        
        # Basic tag based on extension
        ext_tags = {
            ".py": ["code", "python"],
            ".js": ["code", "javascript"],
            ".ts": ["code", "typescript"],
            ".md": ["documentation", "markdown"],
            ".txt": ["text"],
            ".json": ["data", "json"],
            ".yaml": ["config", "yaml"],
            ".yml": ["config", "yaml"],
            ".csv": ["data", "tabular"],
            ".log": ["logs"],
        }
        tags = ext_tags.get(ext.lower(), ["file"])
        
        return GeneratedMetadata(
            summary=summary,
            keywords=keywords,
            tags=tags,
            content_type_hint=ext.lstrip(".") if ext else None
        )
