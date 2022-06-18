# Yolo2Mqtt
Track objects detected by YOLO and publish that info to MQTT

# Description

Yolo2Mqtt is a collection of python scripts for running image recognition on RTSP camera streams using a Yolov5 model,
detecting when two objects interact, saving a video of the event, using MQTT as a communications mechanism.

This program was written to track cat feeding/drinking in a multi-cat home and to graph provide that information to Home Assistant via MQTT. Tracking both the cats and the food/water bowls allowed the bowls
to be moved about the room, rather than designating fixed windows to trigger evnets.

It could be used for pretty much anything in which you want to run YOLO and detect if two objects are overlapping.

## Running

Docker is the easiest way to run this. An external MQTT broker is required as well.
TODO: sample docker-compose

## Configuration

Configuration is done by editing config.yml . All settings and default values (if any) are commented in the default config.yml.

Note: This is meant to be used with a custom-trained YOLOv5 model. The default configuration will download the base yolov5 model,
and the example interactions are some nonsense made with default labels.