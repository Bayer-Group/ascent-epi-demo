"""Turning medical concepts in SQL into real codes.

Holds the placeholder-resolution logic: finding ``[entity@name@vocabulary]``
placeholders in generated SQL, looking the concepts up, and substituting the
resulting codes. The HTTP client for the medical-coder service lives in the
infrastructure layer, so this package stays free of transport concerns.
"""
