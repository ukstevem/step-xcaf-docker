FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb xauth \
    libgl1 libegl1 libglib2.0-0 libx11-6 libxext6 libxrender1 libsm6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole repo so the image includes subpackages (e.g. pipeline/, viewer/, etc.)
# NOTE: add a .dockerignore in your repo to exclude /in, /out, .git, etc.
COPY . /app/

ENTRYPOINT ["python"]
