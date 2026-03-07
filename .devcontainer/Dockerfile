FROM python:3.11-slim

WORKDIR /workspace

# install git
RUN apt-get update && apt-get install -y git

# install poetry
RUN pip install poetry

# copy dependency files first (better caching)
COPY ./pyproject.toml ./README.rst ./

# install dependencies (--no-root to skip installing the package itself, only deps)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root