"""Every field the platform or domain redeclares must match the app's.

RuntimeSettings and DomainSettings exist so the lower layers stop importing
ascent_http.settings. They read the same environment, so a field that appears in
both must agree -- otherwise moving a module silently changes its behaviour in
whatever environment does not set that variable.

This is not hypothetical. Two defaults drifted while writing those classes:

  USER_WAREHOUSE          app "COMPUTE_WH"  vs  None -- would have changed the
                          warehouse for every per-user OAuth connection
  SNOWFLAKE_OAUTH_SCOPE   app ""            vs  None -- MSAL compares scope
                          strings, so None raised TypeError inside the library
                          and broke every MCP tool that needed an OBO token

The first was caught by comparing a handful of fields by hand; the second was
not, and only surfaced when a live tool call failed. Hence this test.
"""

import pytest

import ascent_domain.models.data_definitions  # noqa: F401  (import-cycle order)


def _shared_fields(other):
    from ascent_http.settings import Settings as App

    app = App.model_fields
    return [(n, app[n], f) for n, f in other.model_fields.items() if n in app]


@pytest.mark.parametrize("label", ["runtime", "domain"])
def test_redeclared_fields_keep_the_apps_default(label):
    from ascent_domain.config import DomainSettings
    from ascent_platform.config.runtime import RuntimeSettings

    other = {"runtime": RuntimeSettings, "domain": DomainSettings}[label]
    mismatched = {}
    for name, app_field, own_field in _shared_fields(other):
        # A field the app REQUIRES may be optional lower down -- insisting on
        # credentials is the application's job. The reverse, and any difference
        # in the actual default value, is a behaviour change.
        if app_field.is_required():
            continue
        if own_field.default != app_field.default:
            mismatched[name] = {"app": app_field.default, label: own_field.default}
    assert not mismatched, f"defaults drifted from ascent_http.settings: {mismatched}"
