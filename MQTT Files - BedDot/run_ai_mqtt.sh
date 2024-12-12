#!/bin/bash
clear
CONF=ai_conf.yaml
# CONF="$2"
PYTHON=$(which python3)
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
#echo "Script path: $SCRIPTPATH"
now="$(date +'%d/%m/%Y %T')"

SERVICE="$SCRIPTPATH/main_ai_mqtt.py --conf_file=$CONF"

echo "$SERVICE"

process=$(pgrep -f "$SERVICE")
process=${process[0]}
#echo "the process ID of $mac is $process"

if [[ ! -z $process ]]
then
    echo "$SERVICE is running at $now"
else
    cd $SCRIPTPATH
    $PYTHON $SERVICE
    echo "$SERVICE is restarted at $now"
    # echo "$SERVICE started at $now" >> $SCRIPTPATH/$LOG

fi
