# Start with a basic Python OS version 3.14
FROM python:3.14-slim-bookworm

# Copies the 'uv' tool into this container - This gets uv from the official uv container image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set our working directory for the folder to work in
WORKDIR /code

# Ensure that when commands are run it uses the virtual environment - like setting your PATH so you find the right tools first
ENV PATH="/code/.venv/bin:$PATH"

# Copy dependency management files into the folder
COPY "pyproject.toml" "uv.lock" ".python-version" ./

# Install all dependencies using the locked versions - ie use exactly what's in uv.lock
RUN uv sync --locked

# Copy application code, model and mapping dictionary
COPY "predict.py" "model.bin" "mapping_dict.pkl" ./

# Docker exposes port 9696 to listen at
EXPOSE 9696

# When container starts, run this command:
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]