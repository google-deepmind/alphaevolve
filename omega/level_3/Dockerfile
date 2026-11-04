FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --break-system-packages --no-cache-dir -r requirements.txt

COPY verification.py final_program.py ./

ENTRYPOINT ["python3", "final_program.py"]
