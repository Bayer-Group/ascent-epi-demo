from asgi_correlation_id import correlation_id


def get_correlation_id() -> str:
    return correlation_id.get("-").lower()


def get_correlation_id_short() -> str:
    return get_correlation_id()[:8]
