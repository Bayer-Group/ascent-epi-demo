"""tools_v1.py and tools_experimental.py were a ~93% verbatim fork.

Two servers, the same tools, one copy of the logic each. They had already begun
drifting by hand, which is how a fix lands on one surface and not the other --
the pattern behind several findings in the last review.

The implementation now lives in _tools_shared.py. The @mcp.tool registrations
stay per-surface on purpose: docstrings are the model-facing contract and
legitimately differ, and v1 registers two tools experimental does not.
"""

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[2] / "src/ascent_mcp"
SHARED = {
    "_get_medical_coder",
    "_df_to_serializable",
    "_validate_identifier",
    "_execute_sql_impl",
    "_resolve_non_omop_placeholders",
    "_resolve_omop_placeholders",
    "_get_database_schema_handler",
    "_get_ontology_info_handler",
    "_execute_sql_handler",
    "_list_tables_handler",
    "_describe_table_handler",
    "_get_sample_data_handler",
    "_lookup_medical_codes_handler",
    "_resolve_placeholders_handler",
    "_list_placeholder_concepts_handler",
    "_SAFE_IDENTIFIER_RE",
    "_MAX_RESULT_ROWS",
    "PLACEHOLDER_PATTERN",
    "OMOP_PLACEHOLDER_PATTERN",
    "_NO_CODES_SENTINEL_SOURCE",
}
ADAPTERS = {
    "get_database_schema": "_get_database_schema_handler",
    "execute_sql": "_execute_sql_handler",
    "get_ontology_info": "_get_ontology_info_handler",
    "list_tables": "_list_tables_handler",
    "describe_table": "_describe_table_handler",
    "get_sample_data": "_get_sample_data_handler",
    "lookup_medical_codes": "_lookup_medical_codes_handler",
    "resolve_placeholders": "_resolve_placeholders_handler",
    "list_placeholder_concepts": "_list_placeholder_concepts_handler",
}


def _top_level_names(path):
    names = set()
    for n in ast.parse(path.read_text()).body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            names |= {t.id for t in n.targets if isinstance(t, ast.Name)}
    return names


def test_shared_implementation_is_defined_exactly_once():
    shared_mod = _top_level_names(SRC / "_tools_shared.py")
    missing = SHARED - shared_mod
    assert not missing, f"_tools_shared.py no longer defines {missing}"

    for surface in ("tools_v1.py", "tools_experimental.py"):
        redefined = SHARED & _top_level_names(SRC / surface)
        assert not redefined, (
            f"{surface} redefines shared implementation {redefined} instead of importing it — this is how the two surfaces drifted before"
        )


def test_both_surfaces_register_the_same_core_tools():
    """v1 may add tools; experimental must not carry one v1 lacks, and the
    overlap must not silently shrink."""

    def tools(path):
        return {
            n.name
            for n in ast.parse((SRC / path).read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and any("tool" in ast.dump(d) for d in n.decorator_list)
        }

    v1, ex = tools("tools_v1.py"), tools("tools_experimental.py")
    assert ex <= v1, f"experimental registers tools v1 does not: {sorted(ex - v1)}"
    assert len(ex) >= 9, f"the shared tool surface shrank to {sorted(ex)}"


def test_both_surfaces_use_thin_shared_tool_adapters():
    for surface in ("tools_v1.py", "tools_experimental.py"):
        module = ast.parse((SRC / surface).read_text())
        functions = {node.name: node for node in module.body if isinstance(node, ast.AsyncFunctionDef)}

        for tool_name, handler_name in ADAPTERS.items():
            assert tool_name in functions, f"{surface} no longer defines adapter {tool_name}"
            fn = functions[tool_name]
            body = fn.body[1:] if ast.get_docstring(fn) else fn.body
            assert len(body) == 1, f"{surface}:{tool_name} should be a one-call adapter"
            [stmt] = body
            assert isinstance(stmt, ast.Return), f"{surface}:{tool_name} should directly return the shared handler result"
            assert isinstance(stmt.value, ast.Await), f"{surface}:{tool_name} should await the shared handler"
            call = stmt.value.value
            assert isinstance(call, ast.Call), f"{surface}:{tool_name} should call the shared handler"
            assert isinstance(call.func, ast.Name), f"{surface}:{tool_name} should call a directly imported shared handler"
            assert call.func.id == handler_name, f"{surface}:{tool_name} should delegate to {handler_name}, got {call.func.id}"

            declared = [arg.arg for arg in (*fn.args.args, *fn.args.kwonlyargs) if arg.arg != "self"]
            positional = [arg.id for arg in call.args if isinstance(arg, ast.Name)]
            keyword_names = [kw.arg for kw in call.keywords]

            assert all(isinstance(arg, ast.Name) for arg in call.args), f"{surface}:{tool_name} should forward named positional args only"
            assert all(kw.arg is not None for kw in call.keywords), f"{surface}:{tool_name} should not use **kwargs forwarding"

            if call.keywords:
                assert not call.args, f"{surface}:{tool_name} should not mix positional and keyword forwarding"
                assert keyword_names == declared, (
                    f"{surface}:{tool_name} should forward keywords in signature order {declared}, got {keyword_names}"
                )
                forwarded = [kw.value.id for kw in call.keywords if isinstance(kw.value, ast.Name)]
                assert len(forwarded) == len(call.keywords), f"{surface}:{tool_name} should forward keyword values directly from local names"
                assert forwarded == declared, (
                    f"{surface}:{tool_name} should forward keyword values matching its signature {declared}, got {forwarded}"
                )
            else:
                assert positional == declared, (
                    f"{surface}:{tool_name} should forward positional args in signature order {declared}, got {positional}"
                )
