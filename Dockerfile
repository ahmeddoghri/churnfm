FROM python:3.13-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .
CMD ["python", "-m", "churnfm.eval"]
