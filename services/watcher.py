"""
Watcher service for monitoring RADOS pool changes.

This module provides functionality to monitor a Ceph pool for new or
modified objects and automatically index them.
"""

import logging
import time
from typing import Set, Dict, Optional
from datetime import datetime
import signal
import sys

from core.rados_client import RadosClient
from services.indexer import Indexer

logger = logging.getLogger(__name__)


class Watcher:
    """
    Service for watching RADOS pool and auto-indexing changes.
    
    Monitors a Ceph pool for new or modified objects and triggers
    indexing automatically.
    """
    
    def __init__(
        self,
        rados_client: RadosClient,
        indexer: Indexer,
        poll_interval: int = 60
    ):
        """
        Initialize watcher.
        
        Args:
            rados_client: RADOS client instance
            indexer: Indexer service instance
            poll_interval: Polling interval in seconds
        """
        self.rados_client = rados_client
        self.indexer = indexer
        self.poll_interval = poll_interval
        
        self.known_objects: Dict[str, datetime] = {}
        self.running = False
        
        logger.info(f"Initialized Watcher (poll_interval={poll_interval}s)")
    
    def _get_current_objects(self) -> Dict[str, datetime]:
        """
        Get current objects in pool with their modification times.
        
        Returns:
            Dictionary mapping object_name -> modification_time
        """
        objects_dict = {}
        
        try:
            self.rados_client.ensure_connected()
            object_names = self.rados_client.list_objects()
            
            for object_name in object_names:
                try:
                    _, mtime = self.rados_client.get_object_stat(object_name)
                    objects_dict[object_name] = mtime
                except Exception as e:
                    logger.warning(f"Could not stat object {object_name}: {e}")
            
            return objects_dict
            
        except Exception as e:
            logger.error(f"Failed to get current objects: {e}")
            return {}
    
    def _initialize_known_objects(self) -> None:
        """Initialize the set of known objects."""
        logger.info("Initializing known objects...")
        self.known_objects = self._get_current_objects()
        logger.info(f"Tracking {len(self.known_objects)} objects")
    
    def _check_for_changes(self) -> Dict[str, str]:
        """
        Check for new or modified objects.
        
        Returns:
            Dictionary mapping object_name -> change_type ('new' or 'modified')
        """
        changes = {}
        
        current_objects = self._get_current_objects()
        
        # Check for new or modified objects
        for object_name, mtime in current_objects.items():
            if object_name not in self.known_objects:
                changes[object_name] = 'new'
                logger.info(f"New object detected: {object_name}")
            elif mtime > self.known_objects[object_name]:
                changes[object_name] = 'modified'
                logger.info(f"Modified object detected: {object_name}")
        
        # Update known objects
        self.known_objects = current_objects
        
        return changes
    
    def _handle_changes(self, changes: Dict[str, str]) -> None:
        """
        Handle detected changes by indexing objects.
        
        Args:
            changes: Dictionary of object changes
        """
        if not changes:
            return
        
        logger.info(f"Processing {len(changes)} changed objects")
        
        for object_name, change_type in changes.items():
            try:
                logger.info(f"Indexing {change_type} object: {object_name}")
                self.indexer.index_object(object_name, force_reindex=True)
                
            except Exception as e:
                logger.error(f"Failed to index {object_name}: {e}")
    
    def watch_once(self) -> int:
        """
        Perform a single watch cycle.
        
        Returns:
            Number of changes detected
        """
        changes = self._check_for_changes()
        self._handle_changes(changes)
        return len(changes)
    
    def watch(self, duration: Optional[int] = None) -> None:
        """
        Start watching for changes.
        
        Args:
            duration: Optional duration in seconds (None = infinite)
        """
        logger.info(f"Starting watcher (poll_interval={self.poll_interval}s)")
        
        # Initialize known objects
        self._initialize_known_objects()
        
        # Setup signal handlers
        self.running = True
        
        def signal_handler(sig, frame):
            logger.info("Received stop signal")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        start_time = time.time()
        cycle_count = 0
        total_changes = 0
        
        try:
            while self.running:
                cycle_count += 1
                logger.debug(f"Watch cycle {cycle_count}")
                
                # Check for changes
                num_changes = self.watch_once()
                total_changes += num_changes
                
                # Check duration
                if duration:
                    elapsed = time.time() - start_time
                    if elapsed >= duration:
                        logger.info(f"Duration limit reached ({duration}s)")
                        break
                
                # Sleep until next cycle
                if self.running:
                    time.sleep(self.poll_interval)
            
            logger.info(f"Watcher stopped after {cycle_count} cycles ({total_changes} changes)")
            
        except Exception as e:
            logger.error(f"Watcher error: {e}")
            raise
    
    def watch_daemon(self, log_file: Optional[str] = None) -> None:
        """
        Run watcher as a daemon process.
        
        Args:
            log_file: Optional log file path
        """
        if log_file:
            # Setup file logging
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logging.getLogger().addHandler(file_handler)
            logger.info(f"Logging to file: {log_file}")
        
        logger.info("Starting watcher daemon")
        self.watch(duration=None)
    
    def get_stats(self) -> Dict:
        """
        Get watcher statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "poll_interval_seconds": self.poll_interval,
            "tracked_objects": len(self.known_objects),
            "pool_name": self.rados_client.pool_name,
            "is_running": self.running
        }
