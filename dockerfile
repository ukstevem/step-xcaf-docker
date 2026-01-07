FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb xauth \
    libgl1 libegl1 libglib2.0-0 libx11-6 libxext6 libxrender1 libsm6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py /app/

ENTRYPOINT ["python"]
