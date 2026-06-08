import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Read connection URL from FL_DB_URL — never hardcoded.
db_url = os.environ.get("FL_DB_URL")
if not db_url:
    raise RuntimeError("FL_DB_URL environment variable is not set.")
config.set_main_option("sqlalchemy.url", db_url)

# Autogenerate not used — migrations are written as explicit SQL via op.execute().
target_metadata = None


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
