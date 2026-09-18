"""Outbound clients that speak in domain terms.

These live here rather than in ``ascent_platform`` because they are not
generic transport: ``concept_search`` returns ``MedicalConcept`` and friends.
Platform code must not import the domain, so a client that does belongs on
this side of the line. The generic MSAL/aiohttp plumbing they build on stays
in ``ascent_platform.external.ascent_client``.
"""
