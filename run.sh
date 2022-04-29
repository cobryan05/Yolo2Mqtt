#!/bin/bash
# Run interactionTracker.py and yolo2mqtt.py
(python3 yolo2mqtt.py "$@" & python3 interactionTracker.py "$@")