"""Shared platform layer.

Infrastructure with no domain knowledge: configuration, connectors,
observability. The dependency rule is one-way — ``app -> domain -> platform``.
Nothing in here may import from ``ascent_http``, ``ascent_mcp`` or
``ascent_domain``.
"""
