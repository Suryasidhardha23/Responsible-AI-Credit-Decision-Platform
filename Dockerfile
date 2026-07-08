# ──────────────────────────────────────────────────────────────
# Responsible AI Credit Decision Platform - Production Dockerfile
# Multi-stage build: lean image, non-root user, healthcheck
# ──────────────────────────────────────────────────────────────

FROM python:3.13-slim AS base

# Build metadata
LABEL maintainer="Responsible AI Platform Team"
LABEL description="Enterprise Responsible AI Credit Decision Platform"

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ──────────────────────────────────────────────────────────────
# System dependencies
# ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libpq-dev \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Create non-root application user
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# ──────────────────────────────────────────────────────────────
# Python dependencies (separate layer for caching)
# ──────────────────────────────────────────────────────────────
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ──────────────────────────────────────────────────────────────
# Application code
# ──────────────────────────────────────────────────────────────
COPY --chown=appuser:appgroup . .

# Create required media / static directories and set permissions
RUN mkdir -p media/models media/explanations staticfiles \
    && chown -R appuser:appgroup /app

# Switch to non-root user before running collectstatic
USER appuser

# Collect static assets (requires SECRET_KEY to be set)
RUN SECRET_KEY=docker-build-collect-static python manage.py collectstatic --noinput

# ──────────────────────────────────────────────────────────────
# Runtime
# ──────────────────────────────────────────────────────────────
EXPOSE 8000

# Healthcheck: confirm Django is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/health/ || exit 1

# Gunicorn: 4 workers, 120s timeout for ML-heavy requests
CMD ["gunicorn", \
     "responsible_ai_platform.wsgi:application", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "4", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
