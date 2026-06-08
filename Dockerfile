# Stage 1: builder
FROM python:3.12-alpine AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POETRY_VERSION=2.1.3

RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

WORKDIR /app

COPY pyproject.toml poetry.lock* ./

RUN python -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root --only main

# Stage 2: runtime
FROM python:3.12-alpine AS runtime

COPY --from=builder /opt/venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV POETRY_VERSION=2.1.3
RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

WORKDIR /app

COPY ./app /app/app
COPY alembic.ini pyproject.toml poetry.lock* .env* /app/

EXPOSE 8000

CMD ["sh", "-c", "poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --forwarded-allow-ips '*'"]
