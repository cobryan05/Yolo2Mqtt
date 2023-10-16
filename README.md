# Yolo2Mqtt
Track objects detected by YOLO and publish that info to MQTT

# Description

Yolo2Mqtt is a collection of python scripts for running image recognition on RTSP camera streams using a Yolov5 model,
detecting when two objects interact, saving a video of the event, using MQTT as a communications mechanism.

This program was written to track cat feeding/drinking in a multi-cat home and to graph provide that information to Home Assistant via MQTT. Tracking both the cats and the food/water bowls allowed the bowls
to be moved about the room, rather than designating fixed windows to trigger evnets.

It could be used for pretty much anything in which you want to run YOLO and detect if two objects are overlapping.

## Running

Docker is the easiest way to run this. An MQTT broker is required as well. The sample docker-compose will run an
MQTT broker, but for Home Assistant integration the same broker must be used by HA and Yolo2Mqtt

The minimal commands to get running would are:

>     $ git clone --recursive https://github.com/cobryan05/Yolo2Mqtt.git
>     $ cd Yolo2Mqtt
>     Yolo2Mqtt$ docker build . -t yolo2mqtt
>     Yolo2Mqtt$ docker-compose up


This will build the docker container and run the default configuration.

## Configuration

Configuration is done by editing /config/config.yml. A default config.yml is available as config.defaults.yml. You should configure
the volumes in your docker-compose.yml so that so that your custom config.yml is available to it at /config/config.yml.

All settings and default values (if any) are commented in config.defaults.yml.

Note: This is meant to be used with a custom-trained YOLOv5 model. The default configuration will download the base yolov5 model,
and the example interactions are some nonsense made with default labels. Any useful interactions will require a custom model.

## RTSP Proxy

The Docker container internally runs an RTSP Proxy on port 8554. If a port is forwarded to 8554 on Docker container, then any RTSP camera in the config.yml should be proxied at, eg \<hostIp\>:8554/\<cameraName\>


## Home Assistant Integration

There are two ways in which Yolo2Mqtt can be 'integrated' with Home Assistant.

* Yolo2Mqtt's 'mediaRoot' folder can be set as a 'media_dir' in Home Assistant. This will allow browsing and playback of Yolo2Mqtt recordings via the Home Assistant 'media' browser. (See https://www.home-assistant.io/integrations/media_source/#using-custom-or-additional-media-folders)


* When the homeAssistant 'discoveryEnabled' setting is true in config.yml then event interactions will post to the Home Assistant MQTT Discovery topic (https://www.home-assistant.io/docs/mqtt/discovery/) for autoconfiguration. Events will create new binary_sensors in Home Assistant with names such as
<br><br>
 binary_sensor.[entityPrefix]\_[eventName]\_[cameraName]\_[object1]\_[object2]
<br><br>

 In Home Assistant you can create 'aggregation sensors' to help process further process these events. For example, adding this template sensor to Home Assistant:


>     binary_sensor:
>      - platform: template
>        sensors:
>          cattracker_summary_sammy_eating_food:
>            friendly_name: "Sammy Eating Food"
>            unique_id: "cattracker_summary_sammy_food"
>            value_template: >-
>              {% set action='cateatingfood' %}
>              {% set actor='sammy' %}
>
>              {% set re=action + '[a-z_]+' + actor %}
>              {% set ns = namespace(state='off') %}
>
>              {% for entity in states.binary_sensor if entity|regex_search( re ) and entity.state=='on'%}
>                {% set ns.state = 'on' %}
>              {% endfor %}
>              {{ ns.state }}

will create a new **binary_sensor.cattracker_summary_sammy_eating_food** which will be 'true' if any sensor matches the regex **'sammy[a-z]+cateatingfood'**.
Multiple yolo2mqtt 'events' can be combined into one logical Home Assistant event using tricks like this.
