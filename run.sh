#!/bin/bash
# Run interactionTracker.py and yolo2mqtt.py
python3 yolo2mqtt.py "$@" &
pid1="$!"

python3 interactionTracker.py "$@" &
pid2="$!"

wait $pid1 $pid2