import sys,os
import time
from time import sleep
import json
from collections import deque

import paho.mqtt.client as mqtt
from paho.mqtt.client import CallbackAPIVersion
import threading
# import struct
import queue
from multiprocessing import Process, Queue
import yaml
from influxdb import InfluxDBClient
# import math
from ai_predict import ai_unit_process, setup_args_for_ai
from tls_psk_patch import set_tls_psk_patch
from msg_proc import *
import uuid
import logging
from authenticate import start_authenticate_thread, config_mem_cache
from colorama import Fore, Back, Style
from common import get_config_info_from_file, setup_logger_for_multi_process, logger
# from dot_license import DotLicense
from dot_license_dummy import DotLicense
from read_csv import read_csv_to_nested_dict

DRY_RUN_MODE=False
MULTI_THREAD=False

dot_license=None

broker_psk_id="beddot"
broker_psk="xxx"


debug=False

mqtt_dedicated_receive=None
mqtt_dedicated_pubish=None

# Note: This is a multi-thread Queue
mqtt_msg_queue=queue.Queue()

# Note: This is a multi-process Queue
result_queue=Queue()

args=setup_args_for_ai()
if hasattr(args, 'thread'):
    print(f"thread={args.thread}")
    if args.thread:
        MULTI_THREAD=True
    else:
        MULTI_THREAD=False
if hasattr(args, 'debug'):
    debug=args.debug
    print(f"debug={args.debug}")

class MessageQueueMapping:
    def __init__(self):
        self.msg_queue_dict = {}
        self.rw_lock = threading.RLock()
        
    def add(self, mac, process_id, ai_input_q, ai_output_q, group):
        with self.rw_lock:
            if self.msg_queue_dict.get(mac) is not None:
                print("Duplicate entry. Key '{}' already exists.".format(mac))
            else:
                self.msg_queue_dict[mac] = [process_id, ai_input_q, ai_output_q, group]
                
    def remove(self, mac):
        with self.rw_lock:
            try:
                if mac in self.msg_queue_dict:
                    self.msg_queue_dict.pop(mac)
            except Exception as e:
                pass
                
    def get_ai_input_q(self, mac):
        with self.rw_lock:
            id=self.msg_queue_dict.get(mac)
        return id[1] if id is not None else None
    
    def get_ai_output_q(self, mac):
        with self.rw_lock:
            id=self.msg_queue_dict.get(mac)
        return id[2] if id is not None else None
    
    def get_group(self, mac):
        with self.rw_lock:
            id=self.msg_queue_dict.get(mac)
        return id[3] if id is not None else None
    
    def get_process_id(self, mac):
        with self.rw_lock:
            id=self.msg_queue_dict.get(mac)
        return id[0] if id is not None else None
    
    def get_number_of_queue(self):
        return len(self.msg_queue_dict)

    def terminate_all_process(self):
        with self.rw_lock:
            for mac, id in self.msg_queue_dict.items():
                proc=id[0]
                q=id[1]
                if proc.is_alive():
                    while not q.empty():
                        try:
                            q.get_nowait()  # Non-blocking
                        except queue.Empty:
                            break
                    q.put(None)     # Send a None message to notify the thread to terminate 
                    if MULTI_THREAD:
                        proc.join()
                    else:
                        proc.terminate()

    def update_proc_status(self):
        temp={}
        with self.rw_lock:
            temp=self.msg_queue_dict.copy()

        for mac, id in temp.items():
            proc_id=id[0]
            if not proc_id.is_alive():
                with self.rw_lock:
                    self.msg_queue_dict.pop(mac)

                    




mq_map = MessageQueueMapping()

# This function parses the config file and returns a list of values
def get_config_info():
    config_dict = {}
    with open("ai_com_conf.yaml", "r") as stream:
        config_dict = yaml.safe_load(stream)
    return config_dict

mqtt_publishing_cnt=0
def send_vital_result(group, mac,timestamp, hr, rr, bph, bpl, movement, occupancy, occ_timestamp, alert, alert_timestamp):
    topic="/" + group + "/" + mac + "/vital"
 
    payload ="timestamp=" + str(timestamp)

    if hr != -1:
        payload += "; heartrate=" + str(hr) 
    if rr != -1:
        payload += "; respiratoryrate=" + str(rr) 
    if bph != -1:
        payload += "; systolic=" + str(bph) 
    if bpl != -1:
        payload += "; diastolic=" + str(bpl) 
    
    if movement != -1:
        payload += "; movement=" + str(movement)
    
    for index, oc in enumerate(occupancy):
        payload += "; timestamp=" + str(occ_timestamp + index*10**9) # assuming timestamp in nano seconds
        payload += "; occupancy=" + str(occupancy[index])
    
    if alert != -1:
        payload += "; timestamp=" + str(alert_timestamp) + "; alert=" + str(alert)

    if not DRY_RUN_MODE:
        mqtt_dedicated_pubish.publish(topic, payload, qos=1)
        if debug:
            global mqtt_publishing_cnt
            mqtt_publishing_cnt +=1
            if mqtt_publishing_cnt % 100 ==0:
                print(f"mqtt_publishing_cnt={mqtt_publishing_cnt}")
    else:
        # print(topic)
        pass
    return

#============  MQTT message decode and precess schedule =============
def process_schedule(mqtt_msg_q):
    latest_chk_time=time.monotonic()
    while True:
        #get raw data message 
        try:
            msg=mqtt_msg_q.get() 
        except Exception as e:
            logger(f"process_schedule(), failed to get mqtt message from queue. error={e}")
            sys.exit()

        now=time.monotonic() 
        if now -  latest_chk_time > 30:
            mq_map.update_proc_status()
            latest_chk_time=now  
            
        if None == msg:
            continue

        # extract group name from topic
        group="Unnamed"
        substrings = msg.topic.split("/")
        for substr in substrings[1:]:
            if substr:
                group=substr
                break

        # extact information from the message and put them into a buffer
        mac_addr,timestamp, data_interval, data=parse_beddot_data(msg)
        seismic_data={"timestamp":timestamp, "data_interval":data_interval, "data":data}

        mq_id=mq_map.get_ai_input_q(mac_addr)
        if (None == mq_id): 
            # print(f"dot_license.number_of_devices()={dot_license.number_of_devices()}")
            if mq_map.get_number_of_queue() < dot_license.number_of_devices():
                ai_input_q = Queue()
                    # ai_output_q = Queue()

                # Prepare arguments from command line
                ai_kwargs=vars(args)

                # Prepare settings from database/yaml file/CSV file, which has been unified.
                ai_kwargs["alert_settings"]=config_mem_cache.get_alert_setting_by_mac(mac_addr)
                ai_kwargs["version"]=config_mem_cache.get_monitor_target_by_mac(mac_addr)
                
                # Pass device configurations from CSV for AI developer's customized settings
                configs=config_mem_cache.get_all_conf()
                if "csv_conf" in configs:
                    csv_conf_by_mac=configs["csv_conf"].get(mac_addr)
                    if csv_conf_by_mac:
                        ai_kwargs["csv_conf"]=csv_conf_by_mac
                        # print(f"mac={mac_addr},ai_kwargs[csv_conf]={ai_kwargs['csv_conf']}")

                # Launch a new process/thread
                if MULTI_THREAD:
                    p = threading.Thread(target=ai_unit_process, args=(mac_addr, ai_input_q,result_queue,), kwargs=ai_kwargs, name=mac_addr)
                    p.daemon = True
                else:
                    p = Process(target=ai_unit_process, args=(mac_addr, ai_input_q,result_queue,), kwargs=ai_kwargs, name=mac_addr)
                p.start()

                # Add info to the mapping centre for further use.
                mq_map.add(mac_addr, p, ai_input_q, result_queue, group)
            else:
                logger(f"Number of devices has reached the limit, please contact your provider")

        try:
            if mq_id:
                proc=mq_map.get_process_id(mac_addr)
                backlog=mq_id.qsize()
                if proc.is_alive() and backlog < 180:
                    mq_id.put(seismic_data)
                else:
                    if proc.is_alive():
                        if MULTI_THREAD:
                            # Clear the queue
                            while not mq_id.empty():
                                try:
                                    mq_id.get_nowait()  # Non-blocking
                                except queue.Empty:
                                    break
                            mq_id.put(None)     # Send a None message to notify the thread to terminate 
                        else:
                            proc.terminate()

                        logger(f"Backlog on MAC={mac_addr},{backlog} messages. Process Terminated")
                    mq_map.remove(mac_addr)
                    # logger(f"Task for MAC={mac_addr} Terminated")
        except Exception as e:
            logger(f"Failed to put a raw data message into the queue for mac: {mac_addr}. error={e}")

def publish_result(mqtt_msg_q):

    while True:
        #get result data message 
        try:
            msg=mqtt_msg_q.get() 
        except Exception as e:
            logger(f"process_schedule(), failed to get mqtt message from queue. error={e}")
            sys.exit()
        if None == msg:
            continue

        mac_addr=msg.get("mac_addr")
        if (None == mac_addr):
            logger(f"Missing mac_addr in the message")
            continue
        hr=msg.get("hr",0)   
        rr=msg.get("rr",0)
        bph=msg.get("bph",0)  
        bpl=msg.get("bpl",0)   
        mv=msg.get("mv",0)
        vital_timestamp=msg.get("vital_timestamp",0)   
        oc=msg.get("oc",0)  
        occupancy_timestamp=msg.get("occupancy_timestamp",0)   
        mv=msg.get("mv",0)
        alert=msg.get("alert", -1)   
        alert_timestamp=msg.get("alert_timestamp",0)

        vt=(int(vital_timestamp * 10**9) // 10**7) * 10**7
        oct=(int(occupancy_timestamp * 10**9) // 10**7) * 10**7
        alt=(int(alert_timestamp * 10**9) // 10**7) * 10**7

        group=mq_map.get_group(mac_addr)
        if group:
            send_vital_result(group,mac_addr,vt,hr,rr,bph,bpl, mv, oc, oct, alert, alt)

#============  routine relates to MQTT================================

def on_mqtt_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        logger(f"{Fore.GREEN}Connected to MQTT broker {client._host}: {rc}{Style.RESET_ALL}")
    else:
        logger(f"Connect failed with result code {rc}")
    
    # extract topics
    mqtt_conf=config_mem_cache.get_mqtt()
    
    topics_str=mqtt_conf.get("topic_filter")
    if topics_str is not None:
        # Remove the { } pairs
        topics_str=topics_str.replace("{", "")
        topics_str=topics_str.replace("}", "")
        # Delete the leading and trailing whitespace
        topics_str=topics_str.replace(" ", "")
        # topics_str=topics_str.strip()   
        topics=topics_str.split(",")
        # subscribe topic
        for t in topics:
            if t != "":
                logger(f'subscribe topic: {t}')
                client.subscribe(t,qos=1)

def on_mqtt_message(client, userdata, msg):
    # if (debug): print(f"Received message: {msg.topic}")
    mac=get_mac_from_topic(msg.topic)
    if mac == "" :
        mac=get_mac_from_payload_lagacy(msg.payload)

    if config_mem_cache.is_authentic_mac(mac):
        if (debug): print(f"Authenticated message: {msg.topic}")
        #get message queue id for the dictionary
        mqtt_msg_queue.put(msg)
    return

# def on_mqtt_publish(client, userdata, mid):
#     # print("PUBACK received for message ID:", mid)
#     # increment_msg_ack()
#     return

def on_mqtt_disconnect(client, userdata, flags, rc, properties=None):
    if rc != 0:
        logger(f"MQTT disconnected with error code {rc}. Reason code {rc}")
    else:
        logger("MQTT disconnected successfully")

def init_mqtt(mqtt_conf):
    '''
    mqtt_conf: a dictionary that include:
        mqtt_conf["ip"]: ip address
        mqtt_conf["port"]: prot
        mqtt_conf["user"]: username to login mqtt broker
        mqtt_conf["password"]: password 
        if using CA certificate then the following key-value pair must be defined
        mqtt_conf["ca_cert_path"]:  CA certificate
        mqtt_conf["client_cert_path"]:  Client certificate (optional)
        mqtt_conf["client_key_path"]:   Client key (optional)
    '''
    # Create MQTT client for receiving raw data
    mqtt_port=mqtt_conf.get("port")
    if mqtt_port is None: 
        logger("Invalid MQTT Port, Exit!")
        sys.exit("Invalid MQTT Port, Exit!")
    # print("mqtt_conf=",mqtt_conf)
    unique_client_id = f"AI_Subscribe_{uuid.uuid4()}"
    mqtt_client = mqtt.Client(client_id=unique_client_id, callback_api_version=CallbackAPIVersion.VERSION2)

    if mqtt_port in [7885, 8885, 9885]:
        # Set TLS/SSL options
        try:
            mqtt_client.tls_set(
                ca_certs=mqtt_conf["ca_cert_path"],            # CA certificate
                certfile=mqtt_conf["client_cert_path"],        # Client certificate (optional)
                keyfile=mqtt_conf["client_key_path"]           # Client key (optional)
                # tls_version=ssl.PROTOCOL_TLSv1_2
            )
            mqtt_client.tls_insecure_set(True) # bypass hostname verification, especially for self-signed certificate
        except Exception as e:
            logger(f"Certificate Error={e}")
            sys.exit()
    elif mqtt_port > 8880:
            set_tls_psk_patch(mqtt_client, broker_psk_id, broker_psk)

    if "user" in mqtt_conf:   # check if the broker requires username/password
        pwd = mqtt_conf.get("password", "")
        mqtt_client.username_pw_set(username=mqtt_conf["user"], password=pwd)

    return mqtt_client
    
def setup_mqtt_for_raw_data(mqtt_conf):

    # Create MQTT client for receiving raw data
    mqtt_client=init_mqtt(mqtt_conf)
   
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message
    mqtt_client.on_disconnect = on_mqtt_disconnect
    mqtt_client.connect(mqtt_conf["ip"], mqtt_conf["port"], 60)
    mqtt_thread = threading.Thread(target=lambda: mqtt_client.loop_forever()) 
    mqtt_thread.daemon = True
    mqtt_thread.start()
    return mqtt_client, mqtt_thread


def setup_dedicated_mqtt_for_pubishing_data(mqtt_conf):
    # Create MQTT client for receiving raw data
    mqtt_client=init_mqtt(mqtt_conf)
    
    def on_dedicated_mqtt_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            logger(f"{Fore.GREEN}The dedicated publishing client has successfully connected to the MQTT broker {client._host}: {rc}{Style.RESET_ALL}")
        else:
            logger(f"The dedicated publishing client connection failed: {rc}")
    
    mqtt_client.on_connect = on_dedicated_mqtt_connect

    mqtt_client.connect(mqtt_conf["ip"], mqtt_conf["port"], 60)
    mqtt_thread = threading.Thread(target=lambda: mqtt_client.loop_forever()) 
    mqtt_thread.daemon = True
    mqtt_thread.start()

    return mqtt_client, mqtt_thread

def resolve_path(path, reference_file):
    '''
    path: path need to be resolved
    reference_file: if the 'path' is a relative path then the path to the reference_file serves as the reference path.
    '''
    resolved_path=os.path.expanduser(path)

    if not os.path.isabs(resolved_path):
        # Get the absolute path relative to the yaml_file's directory
        dir = os.path.dirname(reference_file)
        resolved_path = os.path.join(dir, resolved_path)

    # Normalize the path to resolve any '..' or symbolic links
    resolved_path = os.path.realpath(resolved_path)
    # print(f"Absolute path of resolved_path: {resolved_path}")

    return resolved_path


def parse_config_file(yaml_file):

    config_yaml_file=os.path.abspath(os.path.expanduser(yaml_file))

    # Check if the configration file exists
    if not os.path.exists(config_yaml_file):
        sys.exit(f"Error: Config file {config_yaml_file} not found.")
        

    # Create log file based on the command line input
    log_file_name=os.path.splitext(config_yaml_file)[0] + ".log"

    # load configuration from ai_com_conf.yaml
    local_config_info={}
    master_sever=""
    instance_name=""
    instance_scret=""
    config_file_info = get_config_info_from_file(config_yaml_file)
    if "config_mode" in config_file_info:

        if config_file_info["config_mode"] == "local" and "local" in config_file_info:
            local_config_info=config_file_info["local"]
            # Resolve the path of certificates
            if "ca_cert_path" in local_config_info["mqtt"]:
                local_config_info["mqtt"]["ca_cert_path"]=resolve_path(local_config_info["mqtt"]["ca_cert_path"], config_yaml_file)
            if "client_cert_path" in local_config_info["mqtt"]:
                local_config_info["mqtt"]["client_cert_path"]=resolve_path(local_config_info["mqtt"]["client_cert_path"], config_yaml_file)
            if "client_key_path" in local_config_info["mqtt"]:
                local_config_info["mqtt"]["client_key_path"]=resolve_path(local_config_info["mqtt"]["client_key_path"], config_yaml_file)

            # resolve the path of the configuatioin CSV file
            if "devices_conf_csv" in local_config_info:
                local_config_info["devices_conf_csv"]=resolve_path(local_config_info["devices_conf_csv"], config_yaml_file)

        elif config_file_info["config_mode"] == "remote" and "remote" in config_file_info:
            if "master_server" in config_file_info["remote"] and \
                "instance_name" in config_file_info["remote"] and \
                "instance_scret" in config_file_info["remote"]:
                master_sever=config_file_info["remote"]["master_server"]
                instance_name=config_file_info["remote"]["instance_name"]
                instance_scret=config_file_info["remote"]["instance_scret"]
            else:
                sys.exit(f"Missing 'master_server' or 'instance_name' or 'instance_scret' in your yaml file")
        else:
            print(f"'config_mode' error in your yaml file")
    else: 
        sys.exit(f"Missing 'config_mode' in your yaml file")

    if ("license_file" in config_file_info):
        license_file=config_file_info["license_file"]
        license_file=resolve_path(license_file, config_yaml_file)
    else:
         sys.exit(f"Missiing license info")

    return local_config_info, log_file_name, master_sever, instance_name, instance_scret, license_file

def add_dev_info_from_csv(config_info):

    if "devices_conf_csv" in config_info:
        csv_dict=read_csv_to_nested_dict(config_info["devices_conf_csv"], "device_mac")

        #If device_conf_csv is defined, the "devices" and "alert" sections will be ignored
        if csv_dict:    
            config_info["devices"]=[]
            config_info["alert"]={}

            # Use CSV file to configure the devices and it alert settings 
            for mac, row in csv_dict.items():
                target=row.get("monitoring_target", "adult")
                dev_info_list=[mac,target]
                config_info["devices"].append(dev_info_list)

                alert_info_str=row.get("alert_setting")
                alert_info_dict = json.loads(alert_info_str)    # convert the str to a dictionary
                alert_info={mac: alert_info_dict}
                
                config_info["alert"].update(alert_info)

            # Save the configuration from CSV file to config_info
            config_info["csv_conf"]=csv_dict


def is_all_threads_alive(thd_list):
    alive=True
    for t in thd_list:
        if not t.is_alive():
            logger(f"thread {t} terminated")
            alive=False
            break
    return alive

if __name__ == '__main__':
    thread_list=[]

    local_config_info, log_file_name, master_server, instance_name, instance_token, license_file=parse_config_file(args.conf_file)
    add_dev_info_from_csv(local_config_info)

    setup_logger_for_multi_process(log_file_name)
    logger("========= ai_com start ...===========")

    try:
        dot_license=DotLicense(license_file)

        if not local_config_info:   # Remote Mode
            # launch a thread to upate authentic MAC addresses
            path_to_cache=os.path.dirname(log_file_name)
            authenticate_thread=start_authenticate_thread(master_server, instance_name, instance_token, path_to_cache)
            thread_list.append(authenticate_thread)

            # Awaiting configuration has been obtained from remote server or cached encrypted file
            while config_mem_cache.get_source() == "":
                sleep(1)
                if not is_all_threads_alive(thread_list): sys.exit("Threads Error")
        else:
            config_mem_cache.set_all_conf(local_config_info)
            config_mem_cache.set_source("local")

        logger(f"Got configuration from {config_mem_cache.get_source()}")

        # print(f"config_mem_cache = {config_mem_cache.get_all_conf()}")

        # Create MQTT client for receiving raw data
        mqtt_conf=config_mem_cache.get_mqtt()
        mqtt_dedicated_receive, mqtt_thread_recv=setup_mqtt_for_raw_data(mqtt_conf)
        thread_list.append(mqtt_thread_recv)
        # Create a dedicated MQTT client for publishing vital data
        mqtt_dedicated_pubish, mqtt_thread_pub=setup_dedicated_mqtt_for_pubishing_data(mqtt_conf)
        thread_list.append(mqtt_thread_pub)

        # Luanch precess scheduling thread and result publishing thread
        schedule_thread = threading.Thread(target=process_schedule, args=(mqtt_msg_queue,) )
        schedule_thread.daemon = True
        schedule_thread.start()
        thread_list.append(schedule_thread)

        publish_thread = threading.Thread(target=publish_result, args=(result_queue,) )
        publish_thread.daemon = True
        publish_thread.start()
        thread_list.append(publish_thread)

        loop_cnt=0
        while True:
            sleep(1)
            # check if "thread" is still avlive, if no quit the program
            if not is_all_threads_alive(thread_list): break
            loop_cnt=loop_cnt+1
            if loop_cnt % 100 == 1:
                dev_str=f"{mq_map.get_number_of_queue()}/{dot_license.number_of_devices()}"
                expire_str=f"{dot_license.get_expiration_date()}"
                if not dot_license.runtime_verify():
                    logger(f"license checking Failed! [ Devices={dev_str}, Expiration: {expire_str}]")
                    break
                logger(f"Status: active, Devices={dev_str}, Expiration: {expire_str}")
                
    except KeyboardInterrupt:
        logger("Ctrl+C pressed! Cleaning up resources...")
    finally:
        mqtt_dedicated_receive.disconnect()  # Disconnect the MQTT client
        mqtt_dedicated_receive.loop_stop()   # Stop the loop
        mqtt_dedicated_pubish.disconnect()
        mqtt_dedicated_pubish.loop_stop()

        mqtt_thread_recv.join()        # Wait for the thread to finish
        mqtt_thread_pub.join()        # Wait for the thread to finish

        mq_map.terminate_all_process()
        logger("ALL processes terminated!")
        # sleep(3)
        logger("========= ai_com exit ===========")
    sys.exit()


