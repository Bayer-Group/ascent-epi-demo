import logging
import secrets
import time

import snowflake.connector
from fastapi import Request
from opentelemetry import metrics, propagate, trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.botocore import BotocoreInstrumentor

# import instrumentation
from opentelemetry.instrumentation.dbapi import trace_integration
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.propagate import inject
from opentelemetry.propagators.aws import AwsXRayPropagator
from opentelemetry.sdk.extension.aws.trace import AwsXRayIdGenerator
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from starlette.requests import Request as StarletteRequest

from ascent_platform.config.runtime import settings

# Set up AWS X-Ray Propagator
propagate.set_global_textmap(AwsXRayPropagator())

tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

SERVICE_NAME = "ascent-epi-demo"
resource_attributes = {
    "service.name": SERVICE_NAME,
    "aws.log.group.names": [f"{SERVICE_NAME}-{settings.RELEASE_VERSION}"],
}
resource = Resource.create(attributes=resource_attributes)

# Setting up Traces
processor = BatchSpanProcessor(OTLPSpanExporter())
xRayIdGenerator = AwsXRayIdGenerator()
tracer_provider = TracerProvider(resource=resource, active_span_processor=processor, id_generator=xRayIdGenerator)

# apply instrumentation
trace_integration(snowflake.connector, "connect", "snowflake")
BotocoreInstrumentor().instrument()
SQLAlchemyInstrumentor().instrument()
RedisInstrumentor().instrument()
RequestsInstrumentor().instrument()
# WARNING: OpenAI instrumentation may cause compatibility issues with Azure OpenAI in agent routes
# If agent routes fail or hang, comment out the line below to disable OpenAI tracing
# OpenAIInstrumentor().instrument()
AioHttpClientInstrumentor().instrument()

propagator = AwsXRayPropagator()


# Middleware to capture x-amzn-trace-id header automatically
async def capture_amzn_trace_middleware(request: Request, call_next):
    ctx = propagator.extract(request.headers)

    with tracer.start_as_current_span(f"{request.method} {request.url.path}", context=ctx) as span:
        span.set_attribute("request.method", request.method)
        span.set_attribute("request_url_path", request.url.path)
        span.set_attribute("request.url.path", request.url.path)
        span.set_attribute("request.body", getattr(request.state, "body", ""))

        trace_id = span.get_span_context().trace_id
        trace_id_hex = format(trace_id, "032x")
        request.state.trace_id = trace_id_hex

        try:
            response = await call_next(request)
        except Exception as e:
            # Set exception attributes
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR))

            raise e

        # Attach trace_id as attribute and the full amzn-trace-id header
        response.headers["x-trace-id"] = trace_id_hex
        inject(response.headers)

        span.set_attribute("http.status_code", response.status_code)

    return response


def _gen_xray_header() -> str:
    # AWS X-Ray format: Root=1-<epoch-hex>-<24-hex random>;Parent=<16-hex>;Sampled=1
    epoch_hex = f"{int(time.time()):08x}"
    random_96 = secrets.token_hex(12)  # 12 bytes = 24 hex chars
    parent_id = secrets.token_hex(8)  # 8 bytes  = 16 hex chars
    return f"Root=1-{epoch_hex}-{random_96};Parent={parent_id};Sampled=1"


async def add_trace_id(request: Request, call_next):
    # If the client already sent W3C or X-Ray headers, don’t touch it.
    if "x-amzn-trace-id" in request.headers or "traceparent" in request.headers:
        return await call_next(request)

    amzn_header = _gen_xray_header()

    # Starlette headers live in ASGI scope; make a shallow copy and append ours.
    scope = dict(request.scope)
    headers = list(scope.get("headers", []))
    headers.append((b"x-amzn-trace-id", amzn_header.encode("ascii")))
    scope["headers"] = headers

    # Re-wrap the request with the augmented scope so downstream middleware sees it.
    new_request = StarletteRequest(scope, receive=request.receive)

    # Optional: log what we injected so you can grep logs and traces together.
    logging.debug("Injected X-Amzn-Trace-Id: %s", amzn_header)

    response = await call_next(new_request)

    # You already call `inject(response.headers)` later; this is just a safety echo:
    response.headers.setdefault("X-Amzn-Trace-Id", amzn_header)
    return response


def _patch_fastapi_route_details():
    """Guard opentelemetry-instrumentation-fastapi against Starlette 1.0.0.

    Starlette 1.0.0 introduced ``_IncludedRouter``, which has no ``path``
    attribute. ``_get_route_details`` in opentelemetry-instrumentation-fastapi
    (0.61b0) reads ``starlette_route.path`` unguarded on a PARTIAL route match
    (e.g. a CORS preflight OPTIONS), raising ``AttributeError`` in the ASGI
    layer and turning every preflight into a 500 with no CORS headers. Wrap the
    helper to fall back to the raw scope path instead of crashing.
    """
    import opentelemetry.instrumentation.fastapi as otel_fastapi

    if getattr(otel_fastapi._get_route_details, "_ascent_patched", False):
        return

    original_get_route_details = otel_fastapi._get_route_details

    def safe_get_route_details(scope):
        try:
            return original_get_route_details(scope)
        except AttributeError:
            return scope.get("path")

    safe_get_route_details._ascent_patched = True
    otel_fastapi._get_route_details = safe_get_route_details


# FastAPI app instrumentation function
def configure_opentelemetry(app):
    _patch_fastapi_route_details()
    FastAPIInstrumentor.instrument_app(app)
    app.middleware("http")(capture_amzn_trace_middleware)

    trace.set_tracer_provider(tracer_provider)
    # metrics.set_meter_provider(metric_provider)

    logging.info("OpenTelemetry tracing is set to AWS X-Ray.")


def configure_opentelemetry_local(app):
    # Keep the same propagator/ID generator so local == prod semantics
    provider = TracerProvider(id_generator=AwsXRayIdGenerator())
    # provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)

    _patch_fastapi_route_details()
    FastAPIInstrumentor.instrument_app(app)

    # ORDER MATTERS:
    # 1) add_trace_id runs FIRST, injects a synthetic X-Ray header if missing
    # 2) capture_amzn_trace_middleware extracts and starts a span from it
    app.middleware("http")(add_trace_id)
    app.middleware("http")(capture_amzn_trace_middleware)

    logging.info("OpenTelemetry tracing is set to AWS X-Ray (local, with header injection).")
