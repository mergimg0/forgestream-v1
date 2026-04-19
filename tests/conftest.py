import asyncio
from uuid import uuid4

import psycopg
import pytest


TEST_DSN = "postgresql://claude:claude_dev@localhost:5432/forgestream"


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def db_conn():
    conn = await psycopg.AsyncConnection.connect(TEST_DSN, autocommit=False)
    yield conn
    await conn.rollback()
    await conn.close()


@pytest.fixture
def session_id():
    return uuid4()


@pytest.fixture
def branch_id():
    return uuid4()
