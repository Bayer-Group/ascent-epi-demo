from ascent_domain.models.data_definitions import VersionInfo
from ascent_http.settings import settings


def get_version_info() -> VersionInfo:
    # These components ship inside this repo rather than as separately-installed
    # distributions, so they version with the app. The fields stay in the
    # response shape because they are part of the public API contract.
    vendored = f"vendored ({settings.APP_VERSION})"
    return VersionInfo(
        deployment=settings.RELEASE_VERSION,
        version=settings.APP_VERSION,
        ascent_ai_version=vendored,
        ascent_non_omop_version=vendored,
        openai_api_version=settings.OPENAI_API_VERSION,
    )
