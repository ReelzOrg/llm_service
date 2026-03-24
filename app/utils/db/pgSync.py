from typing import Any, cast
from dotenv import load_dotenv
from psycopg import AsyncConnection
from psycopg.rows import dict_row
if not load_dotenv():
  raise Exception("Failed to load .env file")

import os
from contextlib import asynccontextmanager
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

def get_conn_str():
  return f"""
  dbname={os.getenv("POSTGRESQL_DB")}
  user={os.getenv("POSTGRESQL_USER")}
  password={os.getenv("POSTGRESQL_PASSWORD")}
  host={os.getenv("POSTGRESQL_HOST")}
  port={os.getenv("POSTGRESQL_PORT")}
  """ 

# @asynccontextmanager simplifies the management of a resources that require async setup and teardown
@asynccontextmanager
async def get_postgres_checkpointer():
  """
  An async context manager that creates a connection pool,
  sets up the Postgres tables, and yields a checkpointer.
  """

  #Why open=False = https://www.psycopg.org/psycopg3/docs/api/pool.html#the-asyncconnectionpool-class
  async with AsyncConnectionPool(conninfo=get_conn_str(), max_size=10, open=False, kwargs={"row_factory": dict_row}) as pool:
    typed_pool = cast(AsyncConnectionPool[AsyncConnection[dict[str, Any]]], pool)
    checkpointer: AsyncPostgresSaver = AsyncPostgresSaver(typed_pool)

    # Ensure tables exist (Idempotent: safe to run every time on startup)
    # This creates 'checkpoints', 'writes', etc. automatically if missing
    await checkpointer.setup()

    # Yield control back to main.py
    yield checkpointer

"""
CREATE TABLE checkpoint_migrations (
    v INTEGER NOT NULL PRIMARY KEY
);

CREATE TABLE checkpoints (
    thread_id             TEXT    NOT NULL,
    checkpoint_ns         TEXT    NOT NULL DEFAULT '',
    checkpoint_id         TEXT    NOT NULL,
    parent_checkpoint_id  TEXT,
    type                  TEXT,
    checkpoint            JSONB   NOT NULL,
    metadata              JSONB   NOT NULL DEFAULT '{}',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE TABLE checkpoint_blobs (
    thread_id     TEXT    NOT NULL,
    checkpoint_ns TEXT    NOT NULL DEFAULT '',
    channel       TEXT    NOT NULL,
    version       TEXT    NOT NULL,
    type          TEXT    NOT NULL,
    blob          BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);

CREATE TABLE checkpoint_writes (
    thread_id     TEXT    NOT NULL,
    checkpoint_ns TEXT    NOT NULL DEFAULT '',
    checkpoint_id TEXT    NOT NULL,
    task_id       TEXT    NOT NULL,
    task_path     TEXT    NOT NULL,
    idx           INTEGER NOT NULL,
    channel       TEXT    NOT NULL,
    type          TEXT,
    blob          BYTEA   NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);
"""