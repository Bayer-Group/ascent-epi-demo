"""Shared helpers for the MCP tool surfaces.

This package used to hold the tool definitions for the original ``/mcp``
server as well. That server is gone; what remains is the machinery its
successors still share -- permission checks, cohort and CLUES helpers, the
heavy-tool capacity gate and the keepalive/step-budget wrappers.

Deliberately no re-exports. The old ``__init__`` imported every tool module so
that importing the package registered them all with FastMCP, which meant a
single import pulled in the entire surface and its dependencies. Callers now
import the helper they need directly.
"""
