"""The HTTP status codes the frontend and agents actually receive.

FastAPI does not document `raise HTTPException` in the OpenAPI schema, so the
contract snapshot captures only 200/201/202/204/422 -- every 403, 404, 409, 500
and 502 this service returns is invisible to it. Converting these to domain
exceptions with router-level mapping is the next refactor, and a changed status
code is the most likely way it breaks the frontend.

So this pins the surface before the work starts: each exception type, and the
status it produces. The classes will move to ascent_domain and stop inheriting
HTTPException; when they do, this fixture must still describe what the API
returns -- update the LOCATION, never the CODE, unless a status change is the
deliberate point of the commit.
"""

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"

# exception class -> (HTTP status, keys of the detail body the caller receives)
#
# The keys matter as much as the code: the body is a structured dict, so
# `detail.cohort_id` is something a client can read. Values are omitted because
# they are per-instance; the shape is the contract.
ERROR_SURFACE = {
    "CohortAccessDeniedError": (403, ["cohort_id", "error"]),
    "CohortDemographicsUnsupportedError": (422, ["cohort_id", "error"]),
    "CohortNotFoundError": (404, ["cohort_ids", "error"]),
    "CohortNotQueryableError": (409, ["cohort_id", "error"]),
    "CohortTableMissingError": (409, ["cohort_id", "error"]),
    "CannotShareStudyCohortsException": (400, ["error"]),
    "UnknownConceptCodes": (422, ["concept_codes", "error", "vocabulary_id"]),
    "UnknownConceptIDs": (422, ["concept_ids", "error"]),
    "_ExistsError": (403, ["error", "message"]),
    "_IsUsedError": (403, ["error", "message"]),
    "_NotFoundError": (404, ["error", "ids"]),
    "_UnknownError": (422, ["error", "message"]),
}

_CODE_NAMES = {
    "HTTP_400_BAD_REQUEST": 400,
    "HTTP_403_FORBIDDEN": 403,
    "HTTP_404_NOT_FOUND": 404,
    "HTTP_409_CONFLICT": 409,
    # Starlette renamed 422 in 1.0; both names resolve here so this lookup
    # keeps working whichever a module uses. Accessing the old one emits a
    # DeprecationWarning, so first-party code uses CONTENT.
    "HTTP_422_UNPROCESSABLE_CONTENT": 422,
    "HTTP_422_UNPROCESSABLE_ENTITY": 422,
    "HTTP_500_INTERNAL_SERVER_ERROR": 500,
    "HTTP_502_BAD_GATEWAY": 502,
}


def _status_of(node: ast.ClassDef) -> int | None:
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        for kw in sub.keywords:
            if kw.arg != "status_code":
                continue
            v = kw.value
            if isinstance(v, ast.Constant):
                return int(v.value)
            name = getattr(v, "attr", getattr(v, "id", ""))
            if name in _CODE_NAMES:
                return _CODE_NAMES[name]
    return None


def _detail_keys(node: ast.ClassDef) -> list[str]:
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        for kw in sub.keywords:
            if kw.arg == "detail" and isinstance(kw.value, ast.Dict):
                return sorted(k.value for k in kw.value.keys if isinstance(k, ast.Constant))
    return []


def _domain_error_status() -> dict[str, int]:
    """Resolve the status a DomainError subclass maps to via ascent_http.error_mapping."""
    from ascent_http.error_mapping import STATUS_BY_ERROR

    return {cls.__name__: code for cls, code in STATUS_BY_ERROR.items()}


def _declared_errors() -> dict[str, tuple[int, list[str]]]:
    found = {}
    domain_status = _domain_error_status()

    # classes that now inherit a DomainError instead of HTTPException: the code
    # comes from the mapping the router applies, not from the class body
    for path in SRC.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                name = getattr(base, "id", getattr(base, "attr", ""))
                if name in domain_status:
                    found[node.name] = (domain_status[name], _detail_keys(node))
    for path in SRC.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {getattr(b, "id", getattr(b, "attr", "")) for b in node.bases}
            if "HTTPException" not in bases and not (bases & set(ERROR_SURFACE)):
                continue
            status = _status_of(node)
            if status is not None:
                found[node.name] = (status, _detail_keys(node))
    return found


def test_no_error_type_changes_its_status_code():
    """The codes AND the body shape, wherever the classes live."""
    actual = _declared_errors()
    drifted = {
        name: {"was": expected, "now": actual[name]} for name, expected in ERROR_SURFACE.items() if name in actual and actual[name] != expected
    }
    assert not drifted, f"error status or body shape changed -- breaking for callers: {drifted}"


def test_every_pinned_error_type_still_exists():
    """Renaming or deleting one of these silently changes what callers see."""
    actual = _declared_errors()
    missing = sorted(set(ERROR_SURFACE) - set(actual))
    assert not missing, (
        f"error types disappeared: {missing}. If they moved to ascent_domain and "
        f"now map via a router handler, extend _declared_errors() to read that "
        f"mapping -- do not simply drop them from the fixture."
    )


def test_http_exception_handlers_also_catch_domain_errors():
    """`except HTTPException` used to catch the cohort errors. It no longer can.

    Fourteen classes stopped inheriting HTTPException, and every handler that
    caught them by that type silently stopped firing. The suite did not notice:
    clues_ws_service catches them specifically to surface a user-actionable
    message instead of a generic 500, and nothing tests that path. Found by
    reading the catch sites, not by a failure.

    Sites that legitimately catch only HTTPException -- OAuth and ACL paths,
    where nothing raises a DomainError -- are listed explicitly, so adding one
    is a decision rather than an oversight.
    """
    import ast

    HTTP_ONLY = {
        "ascent_domain/database_access.py",  # OBO/ACL failures, HTTPException only
        "ascent_platform/snowflake/session.py",  # ditto, and the platform cannot import the domain
        "ascent_mcp/error_handling.py",  # handles both, in separate clauses
    }

    offenders = []
    for path in SRC.rglob("*.py"):
        # docs/ holds standalone client samples; they are not part of the app
        if "__pycache__" in str(path) or "/docs/" in str(path):
            continue
        rel = str(path.relative_to(SRC))
        if rel in HTTP_ONLY:
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler) or node.type is None:
                continue
            caught = set()
            if isinstance(node.type, ast.Tuple):
                caught = {getattr(e, "id", getattr(e, "attr", "")) for e in node.type.elts}
            else:
                caught = {getattr(node.type, "id", getattr(node.type, "attr", ""))}
            if "HTTPException" in caught and "DomainError" not in caught:
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, f"these catch HTTPException but not DomainError, so domain failures pass through them unhandled: {offenders}"


# Errors raised as themselves rather than subclassed, so the discovery above
# never sees them. They exist because outbound clients used to raise
# HTTPException directly with these codes.
_RAISED_DIRECTLY = {
    "NotAuthenticatedError": 401,
    "UpstreamError": 502,
    "UpstreamUnavailableError": 503,
}


def test_directly_raised_domain_errors_keep_their_status():
    import ascent_domain.errors as errors
    from ascent_http.error_mapping import status_for

    actual = {name: status_for(getattr(errors, name)()) for name in _RAISED_DIRECTLY}
    assert actual == _RAISED_DIRECTLY


def test_upstream_status_error_passes_the_upstream_code_through():
    """An upstream's status reaches the caller verbatim. Collapsing it to a
    flat 502 would hide a 404 or a 429."""
    from ascent_domain.errors import UpstreamStatusError
    from ascent_http.error_mapping import status_for

    for code in (401, 404, 429, 500):
        assert status_for(UpstreamStatusError(code, "upstream said so")) == code
