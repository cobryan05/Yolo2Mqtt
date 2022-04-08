# Start from aniabli/pytorch, no cuda. For GPU try switching to the commented out CUDA version
FROM anibali/pytorch:1.10.2-nocuda
# FROM anibali/pytorch:1.10.2-cuda11.3

WORKDIR /app

RUN git clone https://github.com/cobryan05/Yolo2Mqtt.git --recursive . && \
    pip install -r requirements.txt

ENTRYPOINT ["python", "yolo2mqtt.py", "--config", "/config/config.json"]