FROM python:3.12-slim

WORKDIR /workspace

# install git
RUN apt-get update && apt-get install -y git

# install poetry
RUN pip install poetry

# copy dependency files first (better caching)
COPY ./pyproject.toml ./README.rst ./LICENSE ./

# copy source code
COPY ./src ./src

# install dependencies and the package itself in editable mode
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# install jupyter kernel for VS Code
RUN python3 -m ipykernel install --user --name python3 --display-name "Python 3 (ipykernel)"