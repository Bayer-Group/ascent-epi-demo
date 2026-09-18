"""Business logic that knows nothing about how it is delivered.

Two surfaces sit above this package -- the REST routers in `ascent` and the MCP
tools in `ascent_mcp` -- and before this existed they each reached straight into
services, data access and infrastructure. `ascent_mcp` alone had 123 imports of
`ascent`. That is why a fix so often had to land twice, and why tools_v1 and
tools_experimental could fork without anyone noticing.

The rule, enforced by tests/seams/test_layering_rule.py: nothing here may import
fastapi, fastmcp, starlette, `ascent` or `ascent_mcp`. A module that cannot obey
that is not domain logic yet -- it is still holding a transport concern.

Migration is deliberately partial. The modules here are the ones that were
already free of HTTP. Six others (search, cohort, concept_lists,
clues_ws_service, ai_services, dependencies) still raise HTTPException from
business paths; they move once the routers own that translation.
"""
