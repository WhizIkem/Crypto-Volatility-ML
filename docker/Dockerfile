FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy dependency metadata first so Docker can cache dependency installation.
COPY requirements.txt pyproject.toml README.md ./

RUN python -m pip install --upgrade pip setuptools wheel \
  && python -m pip install -r requirements.txt

# Copy the application source.
COPY . .

# Install the project itself without reinstalling dependencies.
RUN python -m pip install --no-deps -e .

# Safe default:
# building/running the image should not accidentally execute paper trades.
CMD ["python", "-m", "pytest"]