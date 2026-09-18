import glob
import os
from datetime import datetime


def get_latest_querylib_file(main_path):
    """The query library to load.

    Two mechanisms found this file: services.load_query_library reads the
    QUERY_LIBRARY setting, while the RAG agent globbed querylib_*.db out of the
    working directory. They agreed because the deploy dropped the file
    in the CWD under that name. Here the configured path won, the glob found
    nothing, and rag_agent.querylib stayed None -- surfacing much later as
    "'NoneType' object has no attribute 'get_masked_question'".

    The setting wins now, and the glob remains as the fallback.
    """
    try:
        from ascent_platform.config.runtime import get_runtime_settings

        configured = get_runtime_settings().QUERY_LIBRARY
        if configured and os.path.exists(str(configured)):
            return str(configured)
    except Exception:  # noqa: BLE001 - fall back to the historical lookup
        pass

    querylib_files = glob.glob(os.path.join(main_path, "querylib_*.db"))
    querylib_files.sort(
        key=lambda filename: datetime.strptime(filename.split("_")[-1].split(".")[0], "%Y%m%d"),
        reverse=True,
    )
    return querylib_files[0] if querylib_files else None
