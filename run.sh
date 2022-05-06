#!/bin/bash
# Run interactionTracker.py and yolo2mqtt.py
(RTSP_API=TRUE RTSP_APIADDRESS=0.0.0.0:9997 /usr/bin/rtsp-simple-server & \
 python3 yolo2mqtt.py --verbose "$@" & \
 python3 interactionTracker.py "$@" & \
 python3 recordingManager.py --verbose "$@" )