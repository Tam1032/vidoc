import inspect

import pytest
from mcp import Client

from vidoc.mcp_server import build_server
from vidoc.store import ClaimStore


@pytest.fixture
def anyio_backend():
    return "asyncio"


def pytest_collection_modifyitems(items):
    for item in items:
        if inspect.iscoroutinefunction(getattr(item, "function", None)):
            item.add_marker(pytest.mark.anyio)


@pytest.fixture
def store():
    return ClaimStore.load()


@pytest.fixture
async def mcp(store):
    async with Client(build_server(store)) as client:
        yield client
