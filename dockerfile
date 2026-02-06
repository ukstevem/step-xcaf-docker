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

<<<<<<< HEAD
# Copy the whole repo so the image includes subpackages (e.g. pipeline/, viewer/, etc.)
# NOTE: add a .dockerignore in your repo to exclude /in, /out, .git, etc.
COPY . /app/
=======
COPY read_step_xcaf.py /app/read_step_xcaf.py
COPY export_stl_xcaf.py /app/export_stl_xcaf.py
COPY render_thumbnails.py /app/render_thumbnails.py
COPY build_bom_from_xcaf.py /app/build_bom_from_xcaf.py
COPY add_chirality_to_manifest.py /app/add_chirality_to_manifest.py
COPY build_ui_bundle.py /app/build_ui_bundle.py
COPY group_ancillaries.py /app/group_ancillaries.py
COPY build_grouped_ancillaries_summary.py /app/build_grouped_ancillaries_summary.py
>>>>>>> fd53867c1ab2a5e026fd568f264aebd727d4ac2b



ENTRYPOINT ["python"]