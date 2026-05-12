FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps: keep minimal. `psycopg[binary]` does not require libpq-dev.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml

# Optional heavy deps (torch/sentence-transformers) are disabled by default.
ARG INSTALL_RERANK=0
ARG INSTALL_AGENT=1

RUN python -m pip install --upgrade pip \
    && if [ "$INSTALL_AGENT" = "1" ] && [ "$INSTALL_RERANK" = "1" ]; then \
         pip install ".[api,agent,rerank]"; \
       elif [ "$INSTALL_AGENT" = "1" ]; then \
         pip install ".[api,agent]"; \
       elif [ "$INSTALL_RERANK" = "1" ]; then \
         pip install ".[api,rerank]"; \
       else \
         pip install ".[api]"; \
       fi

COPY rag_assistant /app/rag_assistant
COPY scripts /app/scripts
COPY sql /app/sql
COPY data /app/data

EXPOSE 8000

