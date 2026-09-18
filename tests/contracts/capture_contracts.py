"""Capture the externally visible contracts as golden files.

Run with:  PYTHONPATH=./src python tests/contracts/capture_contracts.py
Regenerating is a deliberate act -- the diff is the review artifact.
"""

import ast
import json
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
OUT = pathlib.Path(__file__).parent / "fixtures"


def mcp_contract() -> dict:
    """Tool name -> {server, params, description} for every @mcp.tool.

    Read from the AST rather than a live server: it needs no credentials, no
    event loop, and no FastMCP version pinning, and the decorator plus the
    signature IS the contract.
    """
    out: dict[str, dict] = {}
    for path in sorted(SRC.glob("ascent_mcp/**/*.py")):
        if "__pycache__" in str(path) or "/docs/" in str(path):
            continue
        for fn in ast.parse(path.read_text()).body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in fn.decorator_list:
                dump = ast.dump(dec)
                if ".tool" not in dump and "'tool'" not in dump:
                    continue
                server = next(
                    (c for c in ("mcp_v1", "mcp_experimental", "mcp_ui", "mcp")
                     if f"id='{c}'" in dump), "mcp")
                args = fn.args
                # Defaults are part of the contract, not an implementation
                # detail: an agent that omits an argument gets the default, so
                # changing one changes behaviour for every existing caller.
                # Capturing only name+annotation missed exactly that -- the
                # lookup_medical_codes encoder default went from "gemini" to
                # "bge", pointing every lookup at a different vector index,
                # and this fixture registered no change at all.
                positional = list(args.args)
                pos_defaults = dict(
                    zip([a.arg for a in positional[len(positional) - len(args.defaults):]], args.defaults)
                )
                kw_defaults = {
                    a.arg: d for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None
                }
                defaults = {**pos_defaults, **kw_defaults}
                params = [
                    {
                        "name": a.arg,
                        "annotation": ast.unparse(a.annotation) if a.annotation else None,
                        **({"default": ast.unparse(defaults[a.arg])} if a.arg in defaults else {}),
                    }
                    for a in positional + list(args.kwonlyargs)
                    if a.arg not in ("self", "ctx")
                ]
                out[f"{server}::{fn.name}"] = {
                    "params": params,
                    "description": (ast.get_docstring(fn) or "").strip(),
                }
                break
    return out


def rest_contract(openapi: dict) -> dict:
    """Operation -> params, request model, and the RESOLVED response schema.

    The first version of this fixture stored $ref names. That catches a renamed
    or removed model and nothing else: MetaDataResponse could lose half its
    fields and the test would still pass, which is exactly the change the
    remaining DTO work will make. Resolve one level of $ref so field names and
    types are part of the contract.
    """
    schemas = openapi.get("components", {}).get("schemas", {})

    def resolve(node, depth=0):
        if not isinstance(node, dict) or depth > 3:
            return node
        if "$ref" in node:
            raw = node["$ref"].rsplit("/", 1)[-1]
            target = schemas.get(raw, {})
            name = _strip_variant(raw)
            props = target.get("properties", {})
            return {
                "model": name,
                "required": sorted(target.get("required", [])),
                "fields": {k: _type_of(v) for k, v in sorted(props.items())},
            }
        return node

    def _strip_variant(name: str) -> str:
        """FastAPI emits Model-Input / Model-Output when one model is used in
        both a request and a response. Which variant a given position gets
        depends on how the document was generated, not on what the frontend
        receives, so the suffix is noise for drift detection."""
        for suffix in ("-Input", "-Output"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    def _type_of(v):
        if "$ref" in v:
            return _strip_variant(v["$ref"].rsplit("/", 1)[-1])
        if "anyOf" in v:
            return "|".join(sorted(_type_of(x) for x in v["anyOf"]))
        t = v.get("type", "any")
        if t == "array":
            return f"array[{_type_of(v.get('items', {}))}]"
        return t

    out = {}
    for path, ops in openapi["paths"].items():
        for method, op in ops.items():
            key = f"{method.upper()} {path}"
            out[key] = {
                "params": sorted(p.get("name", "") for p in op.get("parameters", [])),
                "request": resolve(
                    op.get("requestBody", {}).get("content", {})
                      .get("application/json", {}).get("schema", {})),
                "responses": {
                    code: resolve(r.get("content", {}).get("application/json", {}).get("schema", {}))
                    for code, r in op.get("responses", {}).items()
                },
            }
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    mcp = mcp_contract()
    (OUT / "mcp_tools.json").write_text(json.dumps(mcp, indent=2, sort_keys=True) + "\n")
    print(f"captured {len(mcp)} MCP tools")

    import sys

    if len(sys.argv) > 1:  # path to a saved openapi.json
        rest = rest_contract(json.loads(pathlib.Path(sys.argv[1]).read_text()))
        (OUT / "rest_api.json").write_text(json.dumps(rest, indent=2, sort_keys=True) + "\n")
        print(f"captured {len(rest)} REST operations with resolved schemas")
