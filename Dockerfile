# Start from aniabli/pytorch, no cuda. For GPU try switching to the commented out CUDA version
FROM anibali/pytorch:1.10.2-nocuda
# FROM anibali/pytorch:1.10.2-cuda11.3

# Install ffmpeg into the pytorch image
RUN sudo apt-get update && \
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg

# Install python requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy individual apps
COPY --from=aler9/rtsp-simple-server /rtsp-simple-server /usr/bin/rtsp-simple-server

# Now set up this app
WORKDIR /app
COPY . .


ENTRYPOINT ["bash", "./run.sh", "--config", "/config/config.json"]
