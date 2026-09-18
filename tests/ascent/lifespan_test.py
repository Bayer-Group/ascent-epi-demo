import logging

import pytest

from ascent_http.lifespan import FilterDocs


@pytest.mark.asyncio
async def test_filter_docs():
    filter_docs = FilterDocs()
    record = logging.LogRecord(name="", level=logging.INFO, pathname="", lineno=0, msg="/api/docs should be filtered", args=None, exc_info=None)
    assert not filter_docs.filter(record)
    record.msg = "This should not be filtered"
    assert filter_docs.filter(record)
