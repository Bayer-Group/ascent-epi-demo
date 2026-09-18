"""The non-OMOP coder entry point.

The retry policy it carries is the only such policy on any coder path, so it is
worth pinning:

  * transient connect/timeout  -> retried
  * 401                        -> token invalidated, retried
  * 4xx/5xx                    -> fails immediately (retrying amplifies an outage)

and the envelope callers read (``response_data``, ``error``) either way.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from ascent_platform.external import medical_coder_client as mcc


@pytest.fixture
def client():
    """A stub MedicalCoder, with the lru_cache on _client() bypassed."""
    mcc._client.cache_clear()
    stub = MagicMock()
    stub.query_ascent_api = AsyncMock()
    stub.invalidate_token = MagicMock()
    with patch.object(mcc, "_client", return_value=stub):
        yield stub


def _response_error(status: int) -> aiohttp.ClientResponseError:
    return aiohttp.ClientResponseError(request_info=MagicMock(), history=(), status=status, message=f"HTTP {status}")


@pytest.mark.asyncio
async def test_happy_path_returns_the_raw_response_under_response_data(client):
    client.query_ascent_api.return_value = {"response": [{"CONCEPT_ID": 1}]}

    out = await mcc.afetch_medical_coder_full_result(query="diabetes", top_k=5)

    assert out["response_data"] == {"response": [{"CONCEPT_ID": 1}]}
    assert out["query"] == "diabetes"
    assert out["top_k"] == 5
    assert "error" not in out
    assert client.query_ascent_api.await_count == 1


@pytest.mark.asyncio
async def test_the_raw_endpoint_is_used_not_the_reshaping_one(client):
    """get_medical_codes would run construct_response; callers read raw JSON."""
    client.query_ascent_api.return_value = {}

    await mcc.afetch_medical_coder_full_result(query="x")

    endpoint = client.query_ascent_api.await_args.args[0]
    assert endpoint == "get-medical-codes"
    client.get_medical_codes.assert_not_called()


@pytest.mark.asyncio
async def test_transient_connection_error_is_retried(client):
    client.query_ascent_api.side_effect = [
        aiohttp.ClientConnectionError("boom"),
        {"ok": True},
    ]

    with patch.object(mcc, "wait_exponential_jitter", return_value=lambda *_: 0):
        out = await mcc.afetch_medical_coder_full_result(query="x")

    assert out["response_data"] == {"ok": True}
    assert client.query_ascent_api.await_count == 2


@pytest.mark.asyncio
async def test_401_invalidates_the_token_and_retries(client):
    client.query_ascent_api.side_effect = [_response_error(401), {"ok": True}]

    out = await mcc.afetch_medical_coder_full_result(query="x")

    assert out["response_data"] == {"ok": True}
    client.invalidate_token.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 404, 500, 503])
async def test_other_statuses_fail_immediately(client, status):
    """Retrying a struggling coder amplifies the outage."""
    client.query_ascent_api.side_effect = _response_error(status)

    out = await mcc.afetch_medical_coder_full_result(query="x")

    assert out["response_data"] is None
    assert str(status) in out["error"]
    assert client.query_ascent_api.await_count == 1
    client.invalidate_token.assert_not_called()


@pytest.mark.asyncio
async def test_retries_are_bounded(client):
    client.query_ascent_api.side_effect = aiohttp.ClientConnectionError("always")

    out = await mcc.afetch_medical_coder_full_result(query="x")

    assert client.query_ascent_api.await_count == mcc._MAX_ATTEMPTS
    assert out["response_data"] is None
    assert "error" in out


@pytest.mark.asyncio
@pytest.mark.parametrize("domain", ["procedure", "Condition", "MEASUREMENT"])
async def test_domains_not_to_send_are_stripped_case_insensitively(client, domain):
    client.query_ascent_api.return_value = {}

    await mcc.afetch_medical_coder_full_result(query="x", domain_id=domain)

    assert client.query_ascent_api.await_args.kwargs["payload"]["domain_ids"] == []


@pytest.mark.asyncio
async def test_other_domains_are_passed_through(client):
    client.query_ascent_api.return_value = {}

    await mcc.afetch_medical_coder_full_result(query="x", domain_id="Drug")

    assert client.query_ascent_api.await_args.kwargs["payload"]["domain_ids"] == ["Drug"]


@pytest.mark.asyncio
async def test_cosine_similarity_is_omitted_unless_given(client):
    """None must not be sent: the server picks an encoder-specific default,
    and a literal would override it for every encoder."""
    client.query_ascent_api.return_value = {}

    await mcc.afetch_medical_coder_full_result(query="x")
    assert "cosine_similarity" not in client.query_ascent_api.await_args.kwargs["payload"]

    await mcc.afetch_medical_coder_full_result(query="x", cosine_similarity=0.7)
    assert client.query_ascent_api.await_args.kwargs["payload"]["cosine_similarity"] == 0.7


def test_sync_bridge_runs_the_coroutine(client):
    client.query_ascent_api.return_value = {"ok": True}

    out = mcc.execute_medical_coder_full_result(query="x")

    assert out["response_data"] == {"ok": True}


@pytest.mark.asyncio
async def test_sync_bridge_refuses_to_block_a_running_loop():
    """The caller reaches this through asyncio.to_thread. If that hop is ever
    removed, say so rather than deadlock the loop."""
    with pytest.raises(RuntimeError, match="running event loop"):
        mcc.execute_medical_coder_full_result(query="x")


def test_the_non_omop_pipeline_still_reaches_this_through_a_thread():
    """Guards the assumption the sync bridge depends on: sql_postprocessing
    must keep calling it from asyncio.to_thread, not from a coroutine."""
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[2] / "src"
    tree = ast.parse((src / "ascent_domain/non_omop/medical_coding/sql_postprocessing.py").read_text())

    callers = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef)
        for inner in ast.walk(n)
        if isinstance(inner, ast.Call) and getattr(inner.func, "id", "") == "execute_medical_coder_full_result"
    ]
    assert not callers, (
        "execute_medical_coder_full_result is called directly from a coroutine; "
        "it is synchronous and will raise. Await afetch_medical_coder_full_result."
    )
    assert "asyncio.to_thread" in (src / "ascent_domain/non_omop/medical_coding/sql_postprocessing.py").read_text()


# --------------------------------------------------------------------------
# The clinical term must not reach the logs
# --------------------------------------------------------------------------


def test_payload_logging_redacts_the_clinical_term():
    """These are INFO logs on a healthcare service and `query` is the condition
    or drug someone looked up. The shape stays so the log is still useful; the
    term does not."""
    from ascent_platform.external.medical_coder import _loggable

    out = _loggable(
        {
            "query": "metastatic prostate cancer",
            "top_k": 500,
            "vocabulary": ["ICD10"],
            "domain_ids": ["Condition"],
        }
    )
    assert "metastatic prostate cancer" not in str(out)
    assert out["query"] == "<redacted 26 chars>"
    # Everything a reader actually needs is untouched.
    assert out["top_k"] == 500
    assert out["vocabulary"] == ["ICD10"]
    assert out["domain_ids"] == ["Condition"]


def test_code_lists_are_redacted_by_size():
    from ascent_platform.external.medical_coder import _loggable

    out = _loggable({"codes": [1, 2, 3], "database": "SYNTHETIC_EHR_OMOP"})
    assert out["codes"] == "<redacted 3 items>"
    assert out["database"] == "SYNTHETIC_EHR_OMOP"


def test_no_clinical_value_reaches_a_logger():
    """The payload dict is not the only way a term escapes: `logger.error(f"...
    {query} ...")` interpolates the same clinical term directly. Any logger
    argument naming query or codes fails, whether via the payload or on its own.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[2] / "src"
    path = src / "ascent_platform/external/medical_coder.py"
    tree = ast.parse(path.read_text())

    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", None) in {"info", "debug", "warning", "error"}):
            continue
        for arg in node.args:
            rendered = ast.unparse(arg)
            if isinstance(arg, ast.JoinedStr):
                # An f-string may interpolate the term or the whole payload.
                # len(query) is fine -- the length is not the term.
                for name in ("payload", "query", "codes"):
                    if f"{{{name}" in rendered and f"len({name})" not in rendered:
                        offenders.append(f"{node.lineno} (f-string uses {name})")
            elif isinstance(arg, ast.Name) and arg.id in {"payload", "query", "codes"}:
                offenders.append(f"{node.lineno} (passes {arg.id} directly)")
    assert not offenders, f"medical_coder.py lines {offenders} log a clinical value; use _loggable() or log a length instead"
