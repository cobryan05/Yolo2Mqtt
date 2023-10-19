# Start from aniabli/pytorch, no cuda. For GPU try switching to the commented out CUDA version
# FROM anibali/pytorch:1.10.2-nocuda
FROM anibali/pytorch:1.10.2-cuda11.3

# Install ffmpeg into the pytorch image
RUN sudo apt-get update \
    && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg \
    && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y libsm6

# Copy individual apps
COPY --from=bluenviron/mediamtx:1.2.0 /mediamtx /usr/bin/rtsp-simple-server

# Install python requirements
COPY requirements.txt .
COPY submodules/trackerTools/requirements.txt ./requirements2.txt

RUN pip install -r requirements.txt -r requirements2.txt

# Now set up this app
WORKDIR /app
COPY . .
COPY config.defaults.yml /config/config.yml

ENTRYPOINT ["bash", "./run.sh", "--config", "/config/config.yml"]
