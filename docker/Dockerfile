# NOT BUILT OR TESTED YET (no Docker on the development machine). Build from the repo root:
#   docker build -f docker/Dockerfile -t hello-vision:dev .
FROM python:3.12-slim

RUN useradd --create-home --uid 10001 vision
WORKDIR /app

RUN pip install --no-cache-dir uv==0.9.*
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
COPY hello_vision hello_vision
COPY sim sim
RUN uv sync --frozen --no-dev && chown -R vision:vision /app

USER vision
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    VISION_PORT=8080

# Data and models are mounted, never baked into the image (see .env.example for the paths)
VOLUME ["/var/lib/hello-vision"]
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3).status == 200 else 1)"

# The service binds to 127.0.0.1; for container networking set the host in a reverse proxy
# sidecar or change main.py's host deliberately - do not expose the review UI publicly.
CMD ["python", "-m", "hello_vision.service.main"]
