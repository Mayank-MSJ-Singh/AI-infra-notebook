FROM python:3.11-slim

RUN groupadd -r appuser && \
    useradd -m -r -g appuser -u 1000 appuser

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV HOME=/home/appuser
ENV PYTHONUNBUFFERED=1

RUN chown -R appuser:appuser /app /home/appuser

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()"

CMD ["python", "-m", "src.api"]