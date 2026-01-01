FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb xauth \
    libgl1 libegl1 libglib2.0-0 libx11-6 libxext6 libxrender1 libsm6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY read_step_xcaf.py /app/read_step_xcaf.py
COPY export_stl_xcaf.py /app/export_stl_xcaf.py
COPY render_thumbnails.py /app/render_thumbnails.py
COPY build_bom_from_xcaf.py /app/build_bom_from_xcaf.py

ENTRYPOINT ["python"]
