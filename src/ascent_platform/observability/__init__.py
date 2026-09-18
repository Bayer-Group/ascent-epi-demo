"""Tracing and metrics wiring.

OpenTelemetry setup lived in ``ascent/util/otel_setup.py``, inside the FastAPI
app package, though it knows nothing about HTTP routes, cohorts or OMOP -- it
configures exporters and instruments libraries. Its only caller is the app
entrypoint, which is what made it look like app code.
"""

from ascent_platform.observability.otel import (
    configure_opentelemetry,
    configure_opentelemetry_local,
)

__all__ = ["configure_opentelemetry", "configure_opentelemetry_local"]
