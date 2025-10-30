# ---------- Dockerfile ----------
FROM python:3.12-slim

# Basic runtime env
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=America/Phoenix

WORKDIR /app

# OS deps: certificates, timezone, curl (for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata curl \
 && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# App code + templates
COPY app.py ./
COPY templates ./templates

# The app listens on 5000
EXPOSE 5000

# Container-level healthcheck (uses /health endpoint)
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD curl -fsS http://127.0.0.1:5000/health || exit 1

# Gunicorn server (module:function => app:app)
# Tuned for stability + useful logs to stdout/stderr
CMD ["gunicorn","-b","0.0.0.0:5000","--workers","2","--threads","4","--timeout","60","--graceful-timeout","30","--keep-alive","30","--log-level","info","--access-logfile","-","--error-logfile","-","app:app"]

# ---------- end ----------
