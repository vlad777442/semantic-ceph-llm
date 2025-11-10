"""
RADOS Client for Ceph object storage operations.

This module provides a high-level interface to interact with Ceph RADOS,
abstracting connection management, object CRUD operations, and metadata retrieval.
"""

import rados
import logging
from typing import List, Optional, Dict, Tuple, Iterator
from datetime import datetime
import hashlib
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class RadosClient:
    """
    Client for interacting with Ceph RADOS storage.
    
    Provides methods for connecting to Ceph clusters, reading/writing objects,
    and querying object metadata.
    """
    
    def __init__(
        self,
        config_file: str = "/etc/ceph/ceph.conf",
        client_name: str = "client.admin",
        cluster_name: str = "ceph",
        pool_name: str = "cephfs.cephfs.data"
    ):
        """
        Initialize RADOS client.
        
        Args:
            config_file: Path to ceph.conf
            client_name: Ceph client name (e.g., 'client.admin')
            cluster_name: Ceph cluster name
            pool_name: Default pool to operate on
        """
        self.config_file = config_file
        self.client_name = client_name
        self.cluster_name = cluster_name
        self.pool_name = pool_name
        
        self.cluster: Optional[rados.Rados] = None
        self.ioctx: Optional[rados.Ioctx] = None
        self._connected = False
        
        logger.info(f"Initialized RadosClient for pool: {pool_name}")
    
    def connect(self) -> None:
        """
        Establish connection to Ceph cluster.
        
        Raises:
            rados.Error: If connection fails
        """
        try:
            logger.debug(f"Connecting to Ceph cluster with config: {self.config_file}")
            
            self.cluster = rados.Rados(
                conffile=self.config_file,
                name=self.client_name,
                clustername=self.cluster_name
            )
            
            self.cluster.conf_read_file(self.config_file)
            self.cluster.connect()
            
            cluster_id = self.cluster.get_fsid()
            logger.info(f"Connected to Ceph cluster: {cluster_id}")
            
            # Open IO context for the pool
            self.ioctx = self.cluster.open_ioctx(self.pool_name)
            logger.info(f"Opened IO context for pool: {self.pool_name}")
            
            self._connected = True
            
        except rados.Error as e:
            logger.error(f"Failed to connect to Ceph: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close connection to Ceph cluster."""
        if self.ioctx:
            self.ioctx.close()
            logger.debug("Closed IO context")
        
        if self.cluster:
            self.cluster.shutdown()
            logger.info("Disconnected from Ceph cluster")
        
        self._connected = False
    
    @contextmanager
    def connection(self):
        """
        Context manager for automatic connection handling.
        
        Usage:
            with client.connection():
                # perform operations
                pass
        """
        try:
            if not self._connected:
                self.connect()
            yield self
        finally:
            if self._connected:
                self.disconnect()
    
    def ensure_connected(self) -> None:
        """Ensure client is connected, connect if not."""
        if not self._connected:
            self.connect()
    
    def list_pools(self) -> List[str]:
        """
        List all pools in the cluster.
        
        Returns:
            List of pool names
        """
        self.ensure_connected()
        return self.cluster.list_pools()
    
    def get_cluster_stats(self) -> Dict[str, int]:
        """
        Get cluster statistics.
        
        Returns:
            Dictionary with cluster stats (kb, kb_used, kb_avail, num_objects)
        """
        self.ensure_connected()
        return self.cluster.get_cluster_stats()
    
    def list_objects(self, prefix: Optional[str] = None, limit: Optional[int] = None) -> List[str]:
        """
        List objects in the current pool.
        
        Args:
            prefix: Optional prefix filter
            limit: Maximum number of objects to return
            
        Returns:
            List of object names
        """
        self.ensure_connected()
        
        objects = []
        for obj in self.ioctx.list_objects():
            if prefix and not obj.key.startswith(prefix):
                continue
            
            objects.append(obj.key)
            
            if limit and len(objects) >= limit:
                break
        
        logger.debug(f"Listed {len(objects)} objects from pool {self.pool_name}")
        return objects
    
    def object_exists(self, object_name: str) -> bool:
        """
        Check if an object exists in the pool.
        
        Args:
            object_name: Name of the object
            
        Returns:
            True if object exists, False otherwise
        """
        self.ensure_connected()
        
        try:
            self.ioctx.stat(object_name)
            return True
        except rados.ObjectNotFound:
            return False
    
    def get_object_stat(self, object_name: str) -> Tuple[int, datetime]:
        """
        Get object statistics.
        
        Args:
            object_name: Name of the object
            
        Returns:
            Tuple of (size_bytes, modification_time)
            
        Raises:
            rados.ObjectNotFound: If object doesn't exist
        """
        self.ensure_connected()
        
        stat = self.ioctx.stat(object_name)
        size_bytes = stat[0]
        mtime = datetime.fromtimestamp(stat[1].timestamp())
        
        return size_bytes, mtime
    
    def read_object(self, object_name: str, max_size: Optional[int] = None) -> bytes:
        """
        Read object content.
        
        Args:
            object_name: Name of the object
            max_size: Maximum bytes to read (None for all)
            
        Returns:
            Object content as bytes
            
        Raises:
            rados.ObjectNotFound: If object doesn't exist
        """
        self.ensure_connected()
        
        if max_size:
            data = self.ioctx.read(object_name, length=max_size)
        else:
            data = self.ioctx.read(object_name)
        
        logger.debug(f"Read {len(data)} bytes from object: {object_name}")
        return data
    
    def write_object(self, object_name: str, data: bytes) -> None:
        """
        Write data to an object (overwrites existing).
        
        Args:
            object_name: Name of the object
            data: Data to write
        """
        self.ensure_connected()
        
        self.ioctx.write_full(object_name, data)
        logger.debug(f"Wrote {len(data)} bytes to object: {object_name}")
    
    def delete_object(self, object_name: str) -> None:
        """
        Delete an object.
        
        Args:
            object_name: Name of the object
            
        Raises:
            rados.ObjectNotFound: If object doesn't exist
        """
        self.ensure_connected()
        
        self.ioctx.remove_object(object_name)
        logger.debug(f"Deleted object: {object_name}")
    
    def get_xattr(self, object_name: str, attr_name: str) -> Optional[bytes]:
        """
        Get extended attribute from object.
        
        Args:
            object_name: Name of the object
            attr_name: Attribute name
            
        Returns:
            Attribute value or None if not found
        """
        self.ensure_connected()
        
        try:
            return self.ioctx.get_xattr(object_name, attr_name)
        except rados.NoData:
            return None
    
    def set_xattr(self, object_name: str, attr_name: str, attr_value: bytes) -> None:
        """
        Set extended attribute on object.
        
        Args:
            object_name: Name of the object
            attr_name: Attribute name
            attr_value: Attribute value
        """
        self.ensure_connected()
        
        self.ioctx.set_xattr(object_name, attr_name, attr_value)
        logger.debug(f"Set xattr '{attr_name}' on object: {object_name}")
    
    def generate_object_id(self, object_name: str, pool_name: Optional[str] = None) -> str:
        """
        Generate a unique ID for an object.
        
        Args:
            object_name: Name of the object
            pool_name: Pool name (uses default if not provided)
            
        Returns:
            Unique object ID (SHA256 hash)
        """
        pool = pool_name or self.pool_name
        key = f"{pool}:{object_name}"
        return hashlib.sha256(key.encode()).hexdigest()
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False
