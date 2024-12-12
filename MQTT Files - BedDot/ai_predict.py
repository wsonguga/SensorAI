'''
NOTE:
This file is the interface between the framework and the AI module. Which involvs two functions:

1. setup_args_for_ai()
    This function is used to set arguments in the command line. AI developers can add command line arguments to this function as needed.
    All the argruments will be passed to ai_unit_process via "**kwargs". 
    After using "args = argparse.Namespace(**kwargs)" to convert them, developers can access arguments just as they used to.

    This function will be called once during framework startup.

2. ai_unit_process(mac_addr, seismic_data_queue, vital_queue, **kwargs):
    This function will be run as a independent process for a single device. 

    mac_addr, MAC address for a Beddot device.

    seismic_data_queue, is a queue used to recieve seismic data, structured in a dictionary format as shown below. All data has been extracted from the MQTT message.
        seismic_data={
        “timestamp”:			# in nano seconds
        “data_interval”:		# in nano seconds
        “data”:		# a list with data points
        }

    vital_queue, is a queue used to return results from the AI engine to the framework. Messages for the result are structured in a dictionary format as below:
        result={
            "mac_addr": mac_addr,
            "hr":hr,
            "rr":rr,
            "bph":bph,
            "bpl":bpl,
            "mv":mv,
            "vital_timestamp":vital_timestamp,              # in seconds
            "oc":oc,
            "occupancy_timestamp":occupancy_timestamp,      #in seconds
            "alert":alert,                                  # is a number
            "alert_timestamp":alert_timestamp               #in seconds
        }

    **kwargs, settings that are from command line, database, CSV file and Yaml file are passed via this argument.
        --kwargs["command_line_args"], command_line_args is the key word set by parser.add_argument() in the setup_args_for_ai() function
        --kwargs["alert_settings"], the alert setting for "mac_addr". 
        --kwargs["version"], the version setting for "mac_addr". 
        --kwargs["csv_conf"], the original parameter from CSV file. Developers can add fields to the CSV file as needed, which will be passed via this argument.
             The "alert_setting" and "monitoring_target" fields in CSV file are parsed and passed by kwargs["alert_settings"],kwargs["version"]. 
             So, if you don't have additional settings in CSV file, you don't need to access kwargs["csv_conf"]. 

             kwargs["csv_conf"] is a dictionary with MAC address as keyword. e.g.
             ai_kwargs[csv_conf]={'device_id': '305', 'device_mac': '74:4d:bd:89:2d:5c', 'ai_active': '1', 'monitoring_target': 'adult', 'alert_setting': '{ "hr": {"max": 120, "min": 45}, "rr": {"max": 22, "min": 10}, "bph": {"max": 140, "min": 90}, "bpl": {"max": 90, "min": 60}, "oc": {"on": 1, "off": 1} }'}
             

        Use "args = argparse.Namespace(**kwargs)" to convert it to namespace, then use "args.keyword" to access, e.g. args.version

        

Additionally,
1) when setting the device list in Yaml file, '*' can be used as a wildcard, matching all devices, see yaml file. No need to add any code to the framework
2) Define the path to CSV file in Yaml file. You can add or delete columns except "device_mac","monitoring_target","alert_setting"
'''


import numpy as np
from framework_adapter import *
from algo_DSPYS import init_ai_data_buf

import queue
import math
import argparse
from common import logger


def dummy_load():
    j=0
    for i in range(100000):
        j=j+1
        v=i*j
        k=v/45.98
    return(k)

#======================== setting up command line ====================================

def setup_args_for_ai():
    '''
    This function will be called by the framework during startup. 
    All command-line parameters will pass to the ai_unit_process() via kwargs
    '''
    parser = argparse.ArgumentParser(description='BedDot - Sleep Activities and Vital Signs', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # The '-c' argument must be include.
    parser.add_argument('-c','--conf_file', type=str, default='models/ai_conf.yaml',
                    help='the ai yaml conf file') #, required=True)
    
    #parser.add_argument("dot", type=str, help='dot2.us')
    parser.add_argument('--vitals', type=str, default='HRSD', help='the vitals to calculate')
    parser.add_argument('--algo_name', type=str, default='algo_DSPYS', 
                        help='the default algorithm name')
    parser.add_argument('--algo_bp', type=str, default='algo_VTCN', #'algo_LSTMAttention', #
                        help='the default BP model name')
    # parser.add_argument('--debug', type=str, default='False',
    #                     help='the debug mode: True/False')
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction)
    parser.add_argument('--allow_all', action=argparse.BooleanOptionalAction)
    parser.add_argument('--version', type=str, default='adult',
                        help='the algorithm version: adult/animal/baby')
    # parser.add_argument('--rr_duration', type=int, default=45, help='rr duration')
    parser.add_argument('--list_file', type=str, default='', help='the live run list file')
    parser.add_argument('--oc_v', type=str, default='adult_dl', help='the occupancy version: adult_dl/adult_dsp/animal_dsp')
    parser.add_argument('-t','--thread', action=argparse.BooleanOptionalAction)
    parser.add_argument('--bp_path', type=str, default='models/bp_model.onnx', help='the path of bp model')
    parser.add_argument('--encoder_path', type=str, default='models/encoder.onnx', help='the path of oc model encoder')
    parser.add_argument('--clf_path', type=str, default='models/classifier.onnx', help='the path of oc model classifier')
    # parser.add_argument('--enable_license', action=argparse.BooleanOptionalAction, default=True, help="Enable or disable the license feature (default: enabled)")

    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    args.bp_path = os.path.join(base_dir, args.bp_path)
    args.encoder_path = os.path.join(base_dir, args.encoder_path)
    args.clf_path = os.path.join(base_dir, args.clf_path)
    args.conf_file = os.path.join(base_dir, args.conf_file)
    return args

CONSECUTIVE_OCCUPANCY_THRESHOLD = 3
ON_BED = 1
OFF_BED = 0
UNCERTAINTY = -1
debug = False

IGNORE_FIRST = -1
NONE_SENT = 0
LOW_VITAL_ALERT = 1
HIGH_VITAL_ALERT = 2
OFF_BED_ALERT = LOW_VITAL_ALERT
ON_BED_ALERT = HIGH_VITAL_ALERT

MEASURES = ['hr', 'rr', 'bph', 'bpl']
ALERT_CONSECUTIVE_COUNT = 10 #alert_consecutive_count = 10

#======================== Seperated Precess Zone ====================================
# Important Reminder:
# All the functions below are accessed in a seperated process. 
# Therefore, global variable is not available due to the memory isolation with multiprecessing

def predict(args, buffer, timestamp, mac_addr, alert_settings, ai_data_buf):
    # add algorithm details

    if args.version == 'animal':
        args.oc_v = 'animal_dsp'
    # print(ai_data_buf)
    # assuming timestamp in nano seconds
    # timestamp = round(timestamp/10e8, 2)
    init_ai_data_buf(ai_data_buf, args)
    # if "consecutive_occupancy" not in ai_data_buf:
    #     ai_data_buf["consecutive_occupancy"]=0
    #     if(debug): print(f"[{mac_addr}] consecutive_occupancy not in ai_data_buf")

    buffLen = len(buffer)
    hr=rr=sp=dp=movement = -1
    occupancy_timestamp = timestamp
    occupancy_corrected = [-1]
    #################################################################
    # bed occupancy and signal quality
    ################################################################
    occupancy_window = get_occupancy_window(args) 
    if(buffLen >= occupancy_window):
        if(debug): print(f"[{mac_addr}] Calculating bed occupancy and signal quality")
        signal_occupancy = buffer[-occupancy_window:]
        occupancy, params = calc_bed_occupancy(signal_occupancy, ai_data_buf, {'timestamp':timestamp, 'occupancy':[]}, args) 
        # print('occupancy:', occupancy)
        if occupancy == ON_BED: 
            ai_data_buf["consecutive_occupancy"] += 1
            # ai_data_buf["consecutive_occupancy"] = min(ai_data_buf["consecutive_occupancy"], CONSECUTIVE_OCCUPANCY_THRESHOLD)
        else: 
            ai_data_buf["consecutive_occupancy"] = 0

        occupancy_corrected = params['occupancy']
        occupancy_timestamp = params['timestamp']


        if (debug): print(f'[{mac_addr}] nowtime: {timestamp}, occupancy (1: OnBed, 0: OffBed):{occupancy}, consecutive_occupancy:{ai_data_buf["consecutive_occupancy"]} ')

        #################################################################
        # vital signs
        ################################################################
        vital_window = get_vital_window(args)
        # if (debug): print(f'[{mac_addr}] nowtime: {timestamp}, buffLen:{buffLen}, vital_window:{vital_window} ')
        # print('consecutive_occupancy:', ai_data_buf["consecutive_occupancy"])
        # print('buffLen:', buffLen, vital_window)
        # if mac_addr == '30:30:f9:73:4c:34':
        #     print(mac_addr, vital_window, buffLen)
        if (ai_data_buf["consecutive_occupancy"] >= CONSECUTIVE_OCCUPANCY_THRESHOLD and buffLen >= vital_window):
            if(debug): print(f"[{mac_addr}] Calculating vital signs ...")
            signal_vital = buffer[-vital_window:]
            hr, rr, sp, dp, envelope = calc_vital_signs(signal_vital, ai_data_buf, params, args)
            if(debug): print(f'[{mac_addr}] nowtime: {timestamp}, hr: {hr}, rr: {rr}, sp:{sp}, dp: {dp}')                

        #################################################################
        # sleep activities
        ################################################################
        activity_window = get_activity_window(args)
        if (ai_data_buf["consecutive_occupancy"] >= CONSECUTIVE_OCCUPANCY_THRESHOLD and hr == -1 and buffLen >= activity_window):
            if(debug): print(f"[{mac_addr}] Calculating sleep activities ...")
            signal_activity = buffer[-activity_window:]
            movement, params = calc_sleep_activities(signal_activity, params, args) #(signalToMovement, movementThreshold) #, buffertime[len(buffertime)-1], movementShowDelay)
            if movement != 1:
                movement = -1 # movement = -1 will skip sending movement to database to save bandwidth
            if (debug): print(f'[{mac_addr}] nowtime: {timestamp}, movement: {movement}')
            # if movement == 1: write_influx(dst, unit, table_name, 'movement', [movement], timestamp, 1)
            # 'movement'

        # # assuming timestamp in nano seconds
        # occupancy_timestamp = occupancy_timestamp - len(occupancy_corrected)
        # timestamp = round(timestamp*10e8)
        # occupancy_timestamp = round(occupancy_timestamp*10e8)
    mv = movement
    oc = occupancy_corrected
    vital_timestamp = timestamp
    alert, alert_timestamp = check_alert(mac_addr, ai_data_buf, hr,rr,sp,dp,mv,oc, vital_timestamp, alert_settings)
    # alert = -1
    # alert_timestamp = vital_timestamp
    return hr,rr,sp,dp,movement,timestamp, occupancy_corrected, occupancy_timestamp, alert, alert_timestamp


def check_alert(mac_addr, ai_data_buf, hr,rr,sp,dp,mv,oc, vital_timestamp, alert_settings):
    # alert_settings=config_mem_cache.get_alert_setting_by_mac(mac)
    #{ "hr": {"max": 120, "min": 45}, "rr": {"max": 22, "min": 10}, "bph": {"max": 140, "min": 90}, "bpl": {"max": 90, "min": 60}, "oc": {"on": 1, "off": 1} }

    # Check for alerts
    # Put the vital result into the buffer
    vital_window = ai_data_buf["vital_window"]
    # alert_settings = ai_data_buf["alert_settings"]
    alert_sent = ai_data_buf["alert_sent"]
    alert_timestamp = vital_timestamp if ai_data_buf["last_detecting_time"] == -1 else ai_data_buf["last_detecting_time"]

    if oc[0] == -1:
        if ai_data_buf["last_detecting_time"] == -1:
            ai_data_buf["last_detecting_time"] = vital_timestamp
    else:
        ai_data_buf["last_detecting_time"] = -1

    vital_window['hr'].append(hr)
    vital_window['rr'].append(rr)
    vital_window['sp'].append(sp)
    vital_window['dp'].append(dp)
    vital_window['mv'].append(mv)
    vital_window['oc'].append(oc[0])

    if alert_settings is None or alert_settings == '':
        return -1, alert_timestamp

    # Check for occupancy change
    if 'oc' in alert_settings:
        if ('on' in alert_settings['oc'] and alert_settings['oc']['on'] == 1
            and alert_sent['oc'] != ON_BED_ALERT and vital_window['oc'][-1] == 1):

            if alert_sent['oc'] == IGNORE_FIRST:
                alert_sent['oc'] = ON_BED_ALERT
            else:
                alert_sent['oc'] = ON_BED_ALERT
                return 1, alert_timestamp
        elif ('off' in alert_settings['oc'] and alert_settings['oc']['off'] == 1
              and alert_sent['oc'] != OFF_BED_ALERT and vital_window['oc'][-1] == 0):

            if alert_sent['oc'] == IGNORE_FIRST:
                alert_sent['oc'] = OFF_BED_ALERT
            else:
                alert_sent['oc'] = OFF_BED_ALERT
                return 2, alert_timestamp

    # Check for alerts relating to vitals
    for measure in MEASURES:
        over_count = 0
        under_count = 0
        if measure not in alert_settings:
            continue
        vital_measure = measure
        if measure == 'bph':
            vital_measure = 'sp'
        elif measure == 'bpl':
            vital_measure = 'dp'
        # print('measure:', measure)
        if vital_window[vital_measure][-1] > alert_settings[measure]['max']:
            over_count += 1
            i = 2
            while i < len(vital_window[vital_measure]):
                if vital_window[vital_measure][-i] == -1:
                    break
                elif vital_window[vital_measure][-i] > alert_settings[measure]['max']:
                    over_count += 1
                    i += 1
                else:
                    break
        elif vital_window[vital_measure][-1] < alert_settings[measure]['min']:
            under_count += 1
            i = 2
            while i < len(vital_window[vital_measure]):
                if vital_window[vital_measure][-i] == -1:
                    break
                elif vital_window[vital_measure][-i] < alert_settings[measure]['min']:
                    under_count += 1
                    i += 1
                else:
                    break
        
        if over_count >= ALERT_CONSECUTIVE_COUNT and alert_sent[measure] != HIGH_VITAL_ALERT:
            alert_sent[measure] = HIGH_VITAL_ALERT

            return 2 + MEASURES.index(measure)*2 + 1, alert_timestamp
        elif under_count >= ALERT_CONSECUTIVE_COUNT and alert_sent[measure] != LOW_VITAL_ALERT:
            alert_sent[measure] = LOW_VITAL_ALERT

            return 2 + MEASURES.index(measure)*2 + 2, alert_timestamp
        elif over_count < ALERT_CONSECUTIVE_COUNT and under_count < ALERT_CONSECUTIVE_COUNT:
            alert_sent[measure] = NONE_SENT
    # print('alert_sent:', alert_sent)

    # if debug: print(f"ai_unit_proc: {mac_addr}, alert: {res}, {alert_settings}")
    return -1, alert_timestamp

def ai_unit_process(mac_addr, seismic_data_queue, vital_queue, **kwargs):
    
    args = argparse.Namespace(**kwargs)

    # print(f"mac={mac_addr}, alert={args.alert_settings}, version={args.version}")
    
    buffersize   = 60 # config.get('general', 'buffersize')
    samplingrate = 100 # int(config.get('general', 'samplingrate'))
    hrTimeWindow    = 30 # int(config.get('main', 'hrTimeWindow'))
    # BUFFER_SIZE_MAX = int(buffersize) * int(samplingrate)
    # WINDOW_SIZE = elementsNumberHR = hrTimeWindow * samplingrate
    WINDOW_SIZE = max(get_vital_window(args), get_occupancy_window(args))
    BUFFER_SIZE_MAX = WINDOW_SIZE
    raw_data_buf=[]
    ai_data_buf={}

    while True:
        #get raw data message from queue
        try: 
            msg=seismic_data_queue.get(timeout=300) # timeout 5 minutes
        except queue.Empty: #if timeout and does't receive a message, remove mapping dictionary and exit current thread
            logger(f"{mac_addr} have not received message for 5 minute, process terminated")
            break

        if (msg is None) or ("timestamp" not in msg) or ("data_interval" not in msg) or ("data" not in msg):  # If None is received, break the loop
            logger(f"Process {mac_addr}  Received wrong seismic data. exit")
            break
        timestamp=msg["timestamp"]
        data_interval=msg["data_interval"]
        data=msg["data"]
 
        raw_data_buf += data
        buf_len=len(raw_data_buf)
        # dump overflow data
        if(buf_len > BUFFER_SIZE_MAX):
            difSize = buf_len - BUFFER_SIZE_MAX
            del raw_data_buf[0:difSize]

        if buf_len < WINDOW_SIZE :
            continue

        #prep work for AI, and call Ai algrithm
        data = raw_data_buf
        alert_settings=kwargs.get("alert_settings")
        try:
            hr,rr,bph,bpl,mv,vital_timestamp, oc,occupancy_timestamp, alert, alert_timestamp = predict(args, data, math.floor(timestamp/10**9), mac_addr, alert_settings, ai_data_buf)

        except Exception as e:
            logger(f"MAC={mac_addr}: AI predict function ERROR,Terminated: {e}")
            break
        
        result={
            "mac_addr": mac_addr,
            "hr":hr,
            "rr":rr,
            "bph":bph,
            "bpl":bpl,
            "mv":mv,
            "vital_timestamp":vital_timestamp,
            "oc":oc,
            "occupancy_timestamp":occupancy_timestamp,
            "alert":alert,
            "alert_timestamp":alert_timestamp
        }
        try:
            vital_queue.put(result)
        except Exception as e:
            logger(f"MAC={mac_addr}: Send vital ERROR,Terminated: {e}")
            break
    return