"""
LLM-powered agent for natural language Ceph operations.
"""

import logging
import time
from typing import Dict, Any, Optional, List
import json

from core.intent_schema import (
    OperationType, Intent, OperationResult, ConversationHistory
)
from core.llm_provider import BaseLLMProvider
from core.tool_registry import get_all_tools, get_tool_by_name
from core.rados_client import RadosClient
from services.indexer import Indexer
from services.searcher import Searcher
from core.vector_store import VectorStore

logger = logging.getLogger(__name__)


class LLMAgent:
    """
    LLM-powered agent for natural language Ceph storage operations.
    
    Capabilities:
    - Intent classification from natural language
    - Parameter extraction
    - Command execution with validation
    - Natural language response generation
    - Conversational context
    """
    
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        rados_client: RadosClient,
        indexer: Indexer,
        searcher: Searcher,
        vector_store: VectorStore
    ):
        """
        Initialize LLM agent.
        
        Args:
            llm_provider: LLM provider instance
            rados_client: RADOS client for storage operations
            indexer: Indexer service
            searcher: Searcher service
            vector_store: Vector store
        """
        self.llm = llm_provider
        self.rados_client = rados_client
        self.indexer = indexer
        self.searcher = searcher
        self.vector_store = vector_store
        
        self.conversation = ConversationHistory()
        self.tools = get_all_tools()
        
        logger.info("Initialized LLM Agent")
    
    def process_query(self, user_prompt: str, auto_confirm: bool = False) -> OperationResult:
        """
        Main entry point for processing natural language queries.
        
        Args:
            user_prompt: User's natural language input
            auto_confirm: Auto-confirm destructive operations (for non-interactive mode)
            
        Returns:
            OperationResult with execution outcome
        """
        logger.info(f"Processing query: '{user_prompt}'")
        start_time = time.time()
        
        try:
            # Step 1: Classify intent and extract parameters
            intent = self.classify_intent(user_prompt)
            logger.debug(f"Classified intent: {intent.operation} (confidence: {intent.confidence})")
            
            # Step 2: Validate and confirm if needed
            if intent.requires_confirmation and not auto_confirm:
                return OperationResult(
                    success=False,
                    operation=intent.operation,
                    message="Operation requires confirmation",
                    metadata={"intent": intent.to_dict(), "requires_user_confirmation": True}
                )
            
            # Step 3: Execute operation
            result = self.execute_operation(intent)
            result.execution_time = time.time() - start_time
            
            # Step 4: Add to conversation history
            self.conversation.add_message("user", user_prompt)
            self.conversation.add_message("assistant", result.message, {"success": result.success})
            
            return result
        
        except Exception as e:
            logger.error(f"Failed to process query: {e}", exc_info=True)
            return OperationResult(
                success=False,
                operation=OperationType.UNKNOWN,
                error=str(e),
                message=f"Error: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def classify_intent(self, prompt: str) -> Intent:
        """
        Classify user intent and extract parameters.
        
        Args:
            prompt: User's natural language input
            
        Returns:
            Intent object with operation and parameters
        """
        try:
            # Use LLM function calling to determine intent
            system_prompt = """You are a Ceph storage assistant. Analyze the user's request and determine which operation they want to perform."""
            
            result = self.llm.function_call(prompt, self.tools, system=system_prompt)
            
            function_name = result.get('function', 'unknown')
            parameters = result.get('parameters', {})
            reasoning = result.get('reasoning', '')
            
            # Map function name to operation type
            operation = self._map_function_to_operation(function_name)
            
            # Determine if confirmation is needed
            requires_confirmation = operation in [
                OperationType.DELETE_OBJECT,
                OperationType.BULK_DELETE,
                OperationType.UPDATE_OBJECT
            ]
            
            return Intent(
                operation=operation,
                parameters=parameters,
                confidence=0.9,
                reasoning=reasoning,
                requires_confirmation=requires_confirmation,
                original_prompt=prompt
            )
        
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return Intent(
                operation=OperationType.UNKNOWN,
                parameters={},
                confidence=0.0,
                reasoning=str(e),
                requires_confirmation=False,
                original_prompt=prompt
            )
    
    def execute_operation(self, intent: Intent) -> OperationResult:
        """
        Execute the operation specified in the intent.
        
        Args:
            intent: Intent object with operation and parameters
            
        Returns:
            OperationResult
        """
        try:
            operation = intent.operation
            params = intent.parameters
            
            logger.info(f"Executing operation: {operation}")
            
            # Dispatch to appropriate handler
            if operation == OperationType.SEMANTIC_SEARCH:
                return self._handle_search(params)
            
            elif operation == OperationType.READ_OBJECT:
                return self._handle_read(params)
            
            elif operation == OperationType.LIST_OBJECTS:
                return self._handle_list(params)
            
            elif operation == OperationType.CREATE_OBJECT:
                return self._handle_create(params)
            
            elif operation == OperationType.UPDATE_OBJECT:
                return self._handle_update(params)
            
            elif operation == OperationType.DELETE_OBJECT:
                return self._handle_delete(params)
            
            elif operation == OperationType.GET_STATS:
                return self._handle_stats(params)
            
            elif operation == OperationType.INDEX_OBJECT:
                return self._handle_index_object(params)
            
            elif operation == OperationType.BATCH_INDEX:
                return self._handle_batch_index(params)
            
            elif operation == OperationType.FIND_SIMILAR:
                return self._handle_find_similar(params)
            
            elif operation == OperationType.GET_METADATA:
                return self._handle_get_metadata(params)
            
            else:
                return OperationResult(
                    success=False,
                    operation=operation,
                    error="Unknown or unsupported operation",
                    message=f"Operation '{operation}' is not supported"
                )
        
        except Exception as e:
            logger.error(f"Operation execution failed: {e}", exc_info=True)
            return OperationResult(
                success=False,
                operation=intent.operation,
                error=str(e),
                message=f"Error executing {intent.operation}: {str(e)}"
            )
    
    def generate_response(self, result: OperationResult) -> str:
        """
        Generate natural language response from operation result.
        
        Args:
            result: OperationResult
            
        Returns:
            Natural language response string
        """
        if result.message:
            return result.message
        
        # Use LLM to generate friendly response
        try:
            prompt = f"""Generate a friendly, concise response for this operation result:
Operation: {result.operation}
Success: {result.success}
Data: {json.dumps(result.data, indent=2) if result.data else 'None'}
Error: {result.error or 'None'}

Generate a 1-2 sentence natural language response."""
            
            response = self.llm.complete(prompt)
            return response
        
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return result.message or ("Operation completed successfully" if result.success else "Operation failed")
    
    def _map_function_to_operation(self, function_name: str) -> OperationType:
        """Map function name to OperationType."""
        mapping = {
            "search_objects": OperationType.SEMANTIC_SEARCH,
            "read_object": OperationType.READ_OBJECT,
            "list_objects": OperationType.LIST_OBJECTS,
            "create_object": OperationType.CREATE_OBJECT,
            "update_object": OperationType.UPDATE_OBJECT,
            "delete_object": OperationType.DELETE_OBJECT,
            "get_stats": OperationType.GET_STATS,
            "index_object": OperationType.INDEX_OBJECT,
            "batch_index": OperationType.BATCH_INDEX,
            "find_similar": OperationType.FIND_SIMILAR,
            "get_metadata": OperationType.GET_METADATA,
        }
        return mapping.get(function_name, OperationType.UNKNOWN)
    
    # Operation handlers
    
    def _handle_search(self, params: Dict[str, Any]) -> OperationResult:
        """Handle semantic search operation."""
        query = params.get('query', '')
        top_k = params.get('top_k', 10)
        min_score = params.get('min_score', 0.0)
        
        results = self.searcher.search(query, top_k=top_k, min_score=min_score)
        
        if results:
            summary = f"Found {len(results)} objects matching '{query}':\n"
            for i, r in enumerate(results[:5], 1):
                summary += f"{i}. {r.object_name} (score: {r.score:.2f})\n"
            if len(results) > 5:
                summary += f"... and {len(results) - 5} more"
        else:
            summary = f"No objects found matching '{query}'"
        
        return OperationResult(
            success=True,
            operation=OperationType.SEMANTIC_SEARCH,
            data=[r.to_dict() for r in results],
            message=summary
        )
    
    def _handle_read(self, params: Dict[str, Any]) -> OperationResult:
        """Handle read object operation."""
        object_name = params.get('object_name', '')
        
        content = self.rados_client.read_object(object_name)
        
        if content:
            try:
                text = content.decode('utf-8')
                return OperationResult(
                    success=True,
                    operation=OperationType.READ_OBJECT,
                    data={"object_name": object_name, "content": text, "size": len(content)},
                    message=f"Content of '{object_name}':\n{text}"
                )
            except:
                return OperationResult(
                    success=True,
                    operation=OperationType.READ_OBJECT,
                    data={"object_name": object_name, "size": len(content), "binary": True},
                    message=f"Object '{object_name}' contains binary data ({len(content)} bytes)"
                )
        else:
            return OperationResult(
                success=False,
                operation=OperationType.READ_OBJECT,
                error="Object not found or empty",
                message=f"Object '{object_name}' not found"
            )
    
    def _handle_list(self, params: Dict[str, Any]) -> OperationResult:
        """Handle list objects operation."""
        prefix = params.get('prefix')
        limit = params.get('limit', 100)
        
        objects = list(self.rados_client.list_objects(prefix=prefix, limit=limit))
        
        summary = f"Found {len(objects)} objects"
        if prefix:
            summary += f" with prefix '{prefix}'"
        summary += ":\n" + "\n".join(f"- {obj}" for obj in objects[:20])
        if len(objects) > 20:
            summary += f"\n... and {len(objects) - 20} more"
        
        return OperationResult(
            success=True,
            operation=OperationType.LIST_OBJECTS,
            data={"objects": objects, "count": len(objects)},
            message=summary
        )
    
    def _handle_create(self, params: Dict[str, Any]) -> OperationResult:
        """Handle create object operation."""
        object_name = params.get('object_name', '')
        content = params.get('content', '')
        auto_index = params.get('auto_index', True)
        
        data = content.encode('utf-8')
        success = self.rados_client.create_object(object_name, data)
        
        if success:
            message = f"Created object '{object_name}' ({len(data)} bytes)"
            
            # Auto-index if requested
            if auto_index:
                self.indexer.index_object(object_name)
                message += " and indexed for search"
            
            return OperationResult(
                success=True,
                operation=OperationType.CREATE_OBJECT,
                data={"object_name": object_name, "size": len(data)},
                message=message
            )
        else:
            return OperationResult(
                success=False,
                operation=OperationType.CREATE_OBJECT,
                error="Failed to create object",
                message=f"Failed to create object '{object_name}'"
            )
    
    def _handle_update(self, params: Dict[str, Any]) -> OperationResult:
        """Handle update object operation."""
        object_name = params.get('object_name', '')
        content = params.get('content', '')
        append = params.get('append', False)
        
        data = content.encode('utf-8')
        success = self.rados_client.update_object(object_name, data, append=append)
        
        if success:
            action = "Appended to" if append else "Updated"
            return OperationResult(
                success=True,
                operation=OperationType.UPDATE_OBJECT,
                data={"object_name": object_name, "size": len(data)},
                message=f"{action} object '{object_name}'"
            )
        else:
            return OperationResult(
                success=False,
                operation=OperationType.UPDATE_OBJECT,
                error="Failed to update object",
                message=f"Failed to update object '{object_name}'"
            )
    
    def _handle_delete(self, params: Dict[str, Any]) -> OperationResult:
        """Handle delete object operation."""
        object_name = params.get('object_name', '')
        
        success = self.rados_client.delete_object(object_name)
        
        if success:
            return OperationResult(
                success=True,
                operation=OperationType.DELETE_OBJECT,
                data={"object_name": object_name},
                message=f"Deleted object '{object_name}'"
            )
        else:
            return OperationResult(
                success=False,
                operation=OperationType.DELETE_OBJECT,
                error="Failed to delete object",
                message=f"Failed to delete object '{object_name}'"
            )
    
    def _handle_stats(self, params: Dict[str, Any]) -> OperationResult:
        """Handle get stats operation."""
        pool_stats = self.rados_client.get_pool_stats()
        indexer_stats = self.indexer.get_indexing_status()
        vector_stats = self.vector_store.get_stats()
        
        stats = {
            "pool": pool_stats,
            "indexer": indexer_stats,
            "vector_store": vector_stats
        }
        
        message = f"""Storage Statistics:
Pool: {pool_stats.get('num_objects', 0)} objects, {pool_stats.get('size_kb', 0) / 1024:.2f} MB
Indexed: {vector_stats.get('count', 0)} objects
Collection: {vector_stats.get('collection_name', 'unknown')}"""
        
        return OperationResult(
            success=True,
            operation=OperationType.GET_STATS,
            data=stats,
            message=message
        )
    
    def _handle_index_object(self, params: Dict[str, Any]) -> OperationResult:
        """Handle index object operation."""
        object_name = params.get('object_name', '')
        force = params.get('force', False)
        
        metadata = self.indexer.index_object(object_name, force_reindex=force)
        
        if metadata:
            return OperationResult(
                success=True,
                operation=OperationType.INDEX_OBJECT,
                data=metadata.to_dict(),
                message=f"Indexed object '{object_name}'"
            )
        else:
            return OperationResult(
                success=False,
                operation=OperationType.INDEX_OBJECT,
                error="Failed to index object",
                message=f"Failed to index object '{object_name}'"
            )
    
    def _handle_batch_index(self, params: Dict[str, Any]) -> OperationResult:
        """Handle batch index operation."""
        prefix = params.get('prefix')
        limit = params.get('limit')
        force = params.get('force', False)
        
        stats = self.indexer.index_pool(prefix=prefix, limit=limit, force_reindex=force)
        
        message = f"Indexed {stats.indexed_count} objects"
        if stats.skipped_count > 0:
            message += f", skipped {stats.skipped_count}"
        if stats.failed_count > 0:
            message += f", failed {stats.failed_count}"
        
        return OperationResult(
            success=True,
            operation=OperationType.BATCH_INDEX,
            data=stats.to_dict(),
            message=message
        )
    
    def _handle_find_similar(self, params: Dict[str, Any]) -> OperationResult:
        """Handle find similar operation."""
        object_name = params.get('object_name', '')
        top_k = params.get('top_k', 10)
        
        results = self.searcher.find_similar(object_name, top_k=top_k)
        
        if results:
            summary = f"Found {len(results)} objects similar to '{object_name}':\n"
            for i, r in enumerate(results[:5], 1):
                summary += f"{i}. {r.object_name} (similarity: {r.score:.2f})\n"
            if len(results) > 5:
                summary += f"... and {len(results) - 5} more"
        else:
            summary = f"No similar objects found for '{object_name}'"
        
        return OperationResult(
            success=True,
            operation=OperationType.FIND_SIMILAR,
            data=[r.to_dict() for r in results],
            message=summary
        )
    
    def _handle_get_metadata(self, params: Dict[str, Any]) -> OperationResult:
        """Handle get metadata operation."""
        object_name = params.get('object_name', '')
        
        details = self.searcher.get_object_details(object_name)
        
        if details:
            message = f"Metadata for '{object_name}':\n"
            message += f"- Size: {details.get('size_bytes', 0)} bytes\n"
            message += f"- Type: {details.get('content_type', 'unknown')}\n"
            message += f"- Modified: {details.get('modified_at', 'unknown')}"
            
            return OperationResult(
                success=True,
                operation=OperationType.GET_METADATA,
                data=details,
                message=message
            )
        else:
            return OperationResult(
                success=False,
                operation=OperationType.GET_METADATA,
                error="Object not found",
                message=f"Object '{object_name}' not found or not indexed"
            )
    
    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation.clear()
