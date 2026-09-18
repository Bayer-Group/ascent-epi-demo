"""
Custom Redis storage wrapper for MCP OAuth.

This bypasses the buggy RedisStore(client=...) and implements
the AsyncKeyValue protocol expected by FastMCP's encryption wrapper.
"""
import json
from collections.abc import Mapping, Sequence
from typing import Any, SupportsFloat

import redis.asyncio as redis


class RawRedisStorage:
    """
    Simple Redis storage wrapper that properly supports SSL connections.
    
    RedisStore has a bug when using client= parameter - it doesn't write to Redis.
    This wrapper uses redis.asyncio.Redis directly to bypass that issue.
    
    Implements the AsyncKeyValue protocol required by FernetEncryptionWrapper.
    """
    
    def __init__(self, client: redis.Redis):
        """
        Initialize with a pre-configured Redis client.
        
        Args:
            client: Redis client with SSL and other settings configured
        """
        self._client = client
    
    async def get(self, key: str, *, collection: str | None = None) -> dict[str, Any] | None:
        """
        Get a value from Redis.
        
        Args:
            key: The key to retrieve
            collection: Optional collection prefix (ignored - we use full keys)
            
        Returns:
            The deserialized value as a dict or None if not found
        """
        # Prefix key with collection if provided
        full_key = f"{collection}::{key}" if collection else key
        
        value = await self._client.get(full_key)
        if value is None:
            return None
        
        # Deserialize from JSON
        try:
            data = json.loads(value)
            # Always return as dict for protocol compliance
            if not isinstance(data, dict):
                data = {"value": data}
            return data
        except (json.JSONDecodeError, TypeError):
            # If it's not JSON, wrap in dict
            decoded = value.decode() if isinstance(value, bytes) else value
            return {"value": decoded}
    
    async def put(self, key: str, value: Mapping[str, Any], *, collection: str | None = None, ttl: SupportsFloat | None = None) -> None:
        """
        Store a value in Redis.
        
        Args:
            key: The key to store
            value: The mapping value to store (will be JSON serialized)
            collection: Optional collection prefix (ignored - we use full keys)
            ttl: Optional time-to-live in seconds
        """
        # Prefix key with collection if provided
        full_key = f"{collection}::{key}" if collection else key
        
        # Serialize to JSON (convert Mapping to dict for json.dumps)
        serialized = json.dumps(dict(value))
        
        # Set with optional expiration
        if ttl is not None:
            await self._client.set(full_key, serialized, ex=int(float(ttl)))
        else:
            await self._client.set(full_key, serialized)
    
    async def delete(self, key: str, *, collection: str | None = None) -> bool:
        """
        Delete a key from Redis.
        
        Args:
            key: The key to delete
            collection: Optional collection prefix (ignored - we use full keys)
            
        Returns:
            True if the key was deleted, False if it didn't exist
        """
        full_key = f"{collection}::{key}" if collection else key
        result = await self._client.delete(full_key)
        return bool(result)
    
    async def get_many(self, keys: Sequence[str], *, collection: str | None = None) -> list[dict[str, Any] | None]:
        """
        Get multiple values from Redis.
        
        Args:
            keys: List of keys to retrieve
            collection: Optional collection prefix
            
        Returns:
            List of deserialized dicts (None for missing keys)
        """
        if not keys:
            return []
        
        # Prefix keys with collection if provided
        full_keys = [f"{collection}::{key}" if collection else key for key in keys]
        
        values = await self._client.mget(full_keys)
        result = []
        for value in values:
            if value is None:
                result.append(None)
            else:
                try:
                    data = json.loads(value)
                    if not isinstance(data, dict):
                        data = {"value": data}
                    result.append(data)
                except (json.JSONDecodeError, TypeError):
                    decoded = value.decode() if isinstance(value, bytes) else value
                    result.append({"value": decoded})
        return result
    
    async def put_many(self, keys: Sequence[str], values: Sequence[Mapping[str, Any]], *, collection: str | None = None, ttl: SupportsFloat | None = None) -> None:
        """
        Store multiple values in Redis.
        
        Args:
            keys: Sequence of keys to store
            values: Sequence of values to store (must match length of keys)
            collection: Optional collection prefix
            ttl: Optional time-to-live in seconds (applied to all keys)
        """
        if not keys:
            return
        
        # Prefix keys with collection and serialize all values
        serialized = {}
        for key, value in zip(keys, values):
            full_key = f"{collection}::{key}" if collection else key
            serialized[full_key] = json.dumps(dict(value))
        
        # If TTL is specified, set each key individually with expiration
        if ttl is not None:
            ttl_seconds = int(float(ttl))
            for key, value in serialized.items():
                await self._client.set(key, value, ex=ttl_seconds)
        else:
            await self._client.mset(serialized)
    
    async def delete_many(self, keys: Sequence[str], *, collection: str | None = None) -> int:
        """
        Delete multiple keys from Redis.
        
        Args:
            keys: List of keys to delete
            collection: Optional collection prefix
            
        Returns:
            Number of keys deleted
        """
        if not keys:
            return 0
        
        # Prefix keys with collection if provided
        full_keys = [f"{collection}::{key}" if collection else key for key in keys]
        result = await self._client.delete(*full_keys)
        return int(result)
    
    async def ttl(self, key: str, *, collection: str | None = None) -> tuple[dict[str, Any] | None, float | None]:
        """
        Get the value and time-to-live for a key.
        
        Args:
            key: The key to check
            collection: Optional collection prefix
            
        Returns:
            Tuple of (value, ttl_in_seconds) or (None, None) if key doesn't exist
        """
        full_key = f"{collection}::{key}" if collection else key
        
        # Get both value and TTL using pipeline for efficiency
        pipe = self._client.pipeline()
        pipe.get(full_key)
        pipe.ttl(full_key)
        results = await pipe.execute()
        
        raw_value, ttl_seconds = results
        
        if raw_value is None:
            return (None, None)
        
        # Deserialize value
        try:
            value = json.loads(raw_value)
            if not isinstance(value, dict):
                value = {"value": value}
        except (json.JSONDecodeError, TypeError):
            decoded = raw_value.decode() if isinstance(raw_value, bytes) else raw_value
            value = {"value": decoded}
        
        # Convert TTL: -1 = no expiry, -2 = doesn't exist
        ttl_float = None if ttl_seconds < 0 else float(ttl_seconds)
        
        return (value, ttl_float)
    
    async def ttl_many(self, keys: Sequence[str], *, collection: str | None = None) -> list[tuple[dict[str, Any] | None, float | None]]:
        """
        Get values and TTLs for multiple keys.
        
        Args:
            keys: List of keys to check
            collection: Optional collection prefix
            
        Returns:
            List of (value, ttl) tuples
        """
        if not keys:
            return []
        
        # Prefix keys with collection if provided
        full_keys = [f"{collection}::{key}" if collection else key for key in keys]
        
        # Get all values and TTLs using pipeline
        pipe = self._client.pipeline()
        for key in full_keys:
            pipe.get(key)
            pipe.ttl(key)
        results = await pipe.execute()
        
        # Process results in pairs (value, ttl)
        output = []
        for i in range(0, len(results), 2):
            raw_value = results[i]
            ttl_seconds = results[i + 1]
            
            if raw_value is None:
                output.append((None, None))
            else:
                try:
                    value = json.loads(raw_value)
                    if not isinstance(value, dict):
                        value = {"value": value}
                except (json.JSONDecodeError, TypeError):
                    decoded = raw_value.decode() if isinstance(raw_value, bytes) else raw_value
                    value = {"value": decoded}
                
                ttl_float = None if ttl_seconds < 0 else float(ttl_seconds)
                output.append((value, ttl_float))
        
        return output
    
    async def close(self) -> None:
        """Close the Redis connection."""
        await self._client.aclose()
