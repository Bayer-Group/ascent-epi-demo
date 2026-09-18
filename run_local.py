"""Local development server launcher.

Pre-imports torch to avoid DLL initialization failures on Windows
when uvicorn loads the app module in a subprocess.
"""

import torch  # noqa: F401 — pre-load torch DLLs before uvicorn

import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=False)
