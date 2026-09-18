from typing import Any, Optional

import redis.asyncio as redis

# import Cache and cached here to enforce setting configs when using them in other modules
from aiocache import caches
from aiocache.backends.redis import RedisBackend
from aiocache.serializers import JsonSerializer

from ascent_platform.config.runtime import get_runtime_settings

_NOT_SET = object()


class CustomRedisBackend(RedisBackend):
    def __init__(
        self,
        ssl: bool = False,
        ssl_keyfile: Optional[str] = None,
        ssl_certfile: Optional[str] = None,
        ssl_cert_reqs: str = "required",
        ssl_ca_certs: Optional[str] = None,
        ssl_ca_data: Optional[str] = None,
        ssl_check_hostname: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.ssl = ssl
        self.ssl_keyfile = ssl_keyfile
        self.ssl_certfile = ssl_certfile
        self.ssl_cert_reqs = ssl_cert_reqs
        self.ssl_ca_certs = ssl_ca_certs
        self.ssl_ca_data = ssl_ca_data
        self.ssl_check_hostname = ssl_check_hostname

        # NOTE: decoding can't be controlled on API level after switching to
        # redis, we need to disable decoding on global/connection level
        # (decode_responses=False), because some of the values are saved as
        # bytes directly, like pickle serialized values, which may raise an
        # exception when decoded with 'utf-8'.
        # Bounded socket + pool. aiocache defaults `create_connection_timeout`
        # and `pool_max_size` to None and never passes `socket_timeout` at all,
        # so an established connection could block on read indefinitely. Its
        # only guard is an `asyncio.wait_for` around the call, which cancels the
        # coroutine but cannot free the socket — the connection goes back to the
        # pool mid-command and poisons the next caller. This cache sits on the
        # JWKS lookup in `get_public_key`, i.e. the auth path of every MCP
        # request including `initialize`, so a wedged read stalls new sessions.
        self.client: Any = redis.Redis(
            host=self.endpoint,
            port=self.port,
            db=self.db,
            password=self.password,
            decode_responses=False,
            socket_connect_timeout=self.create_connection_timeout or 2,
            socket_timeout=get_runtime_settings().CACHE_REDIS_SOCKET_TIMEOUT,
            max_connections=self.pool_max_size or get_runtime_settings().CACHE_REDIS_MAX_CONNECTIONS,
            health_check_interval=30,
            ssl=self.ssl,
            ssl_keyfile=self.ssl_keyfile,
            ssl_certfile=self.ssl_certfile,
            ssl_cert_reqs=self.ssl_cert_reqs,
            ssl_ca_certs=self.ssl_ca_certs,
            ssl_ca_data=self.ssl_ca_data,
            ssl_check_hostname=self.ssl_check_hostname,
        )


class CustomRedisCache(CustomRedisBackend):
    NAME = "redis"

    def __init__(self, serializer: Any = None, **kwargs: Any) -> None:
        super().__init__(serializer=serializer or JsonSerializer(), **kwargs)

    @classmethod
    def parse_uri_path(cls, path: str) -> dict:
        options = {}
        db, *_ = path[1:].split("/")
        if db:
            options["db"] = db
        return options

    def __repr__(self) -> str:  # pragma: no cover
        return f"RedisCache ({self.endpoint}:{self.port})"


PICKLE_SERIALIZER = "aiocache.serializers.PickleSerializer"

cache_configs = {
    "default": {
        "cache": "aiocache.SimpleMemoryCache",
        "serializer": {"class": PICKLE_SERIALIZER},
    },
    "redis": {
        "cache": "ascent_platform.cache.aiocache_backend.CustomRedisCache",
        "endpoint": get_runtime_settings().REDIS_HOST,
        "port": get_runtime_settings().REDIS_PORT,
        "timeout": 60,
        "ssl": True,
        "ssl_cert_reqs": "none",
        "ssl_check_hostname": False,
        "serializer": {"class": PICKLE_SERIALIZER},
        "plugins": [{"class": "aiocache.plugins.HitMissRatioPlugin"}, {"class": "aiocache.plugins.TimingPlugin"}],
    },
    "redis-local": {
        "cache": "ascent_platform.cache.aiocache_backend.CustomRedisCache",
        "endpoint": get_runtime_settings().REDIS_HOST,
        "port": get_runtime_settings().REDIS_PORT,
        "timeout": 60,
        "ssl": False,
        "serializer": {"class": PICKLE_SERIALIZER},
        "plugins": [{"class": "aiocache.plugins.HitMissRatioPlugin"}, {"class": "aiocache.plugins.TimingPlugin"}],
    },
}

caches.set_config(cache_configs)

cache = caches.create(get_runtime_settings().CACHE)
