
from scipy.signal import butter, lfilter
from scipy import signal
from datetime import datetime, date
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
import numpy
import random
import time
import operator
import sys, os
import logging
from fastsst.sst import *
#from scipy import stats
import nitime.algorithms as nt_alg
import numpy as np
from numpy import array
import scipy as sp
import threading
from datetime import datetime
from dateutil import tz
import pytz
import warnings
import ast
import requests
import subprocess
from dateutil.parser import parse
#from config import Config
import webbrowser
from util import * 

warnings.filterwarnings("ignore")

#influxdb config
token = "0ML4vBa-81dGKI3_wD-ReiSRdLggdJPXKoTKLPITBcOZXl8MJh7W8wFSkNUNM_uPS9mJpzvBxUKfKgie0dHiow=="
org = "lab711"
bucket = "testbed"
url = "sensorwebdata.engr.uga.edu:8086"
measurement = "detection_results"

"""
client = influxdb_client.InfluxDBClient(
    url=url,
    token=token,
    org=org
)
#"""

# rip = ip
debug = True; #str2bool(config.get('general', 'debug'))
verbose = True

# src = {'ip': 'https://sensorweb.us', 'db': 'shake', 'user':'test', 'passw':'sensorweb'}
# dest = {'ip': 'https://sensorweb.us', 'db': 'algtest', 'user':'test', 'passw':'sensorweb'}

# src = {'ip': 'https://sensorwebdata.engr.uga.edu', 'db': 'satcdb', 'user':'test', 'passw':'sensorweb128'}
# dest = {'ip': 'https://sensorwebdata.engr.uga.edu', 'db': 'satcdb', 'user':'test', 'passw':'sensorweb128'}

src = {'ip': 'http://sensorwebdata.engr.uga.edu', 'org':'lab711','token':'0ML4vBa-81dGKI3_wD-ReiSRdLggdJPXKoTKLPITBcOZXl8MJh7W8wFSkNUNM_uPS9mJpzvBxUKfKgie0dHiow==','bucket':'testbed'}
dest = {'ip': 'http://sensorwebdata.engr.uga.edu', 'org':'lab711','token':'0ML4vBa-81dGKI3_wD-ReiSRdLggdJPXKoTKLPITBcOZXl8MJh7W8wFSkNUNM_uPS9mJpzvBxUKfKgie0dHiow==','bucket':'testbed'}

def str2bool(v):
  return v.lower() in ("true", "1", "https", "t")

########### main entrance ########
def main():
 progname = sys.argv[0]
 if(len(sys.argv)<2):
    print("Usage: %s mac [start] [end] [ip] [https/http]" %(progname))
    print("Example: %s b8:27:eb:97:f5:ac   # start with current time and run in real-time as if in a node" %(progname))
    print("Example: %s b8:27:eb:97:f5:ac 2020-08-13T02:03:00.200 # start with the specified time and run non-stop" %(progname))
    print("Example: %s b8:27:eb:97:f5:ac 2020-08-13T02:03:00.200 2020-08-13T02:05:00.030 # start and end with the specified time" %(progname))
    print("Example: %s b8:27:eb:97:f5:ac 2020-08-13T02:03:00.200 2020-08-13T02:05:00.030 sensorweb.us https # specify influxdb IP and http/https" %(progname))
    quit()

 # Parameters from Config file
 db           = bucket #'satcdb' # config.get('general', 'dbraw')
 buffersize   = 30 # config.get('general', 'buffersize')
 samplingrate = 2 # int(config.get('general', 'samplingrate'))
 hrTimeWindow    = 30 # int(config.get('main', 'hrTimeWindow'))
 maxbuffersize   = int(buffersize) * int(samplingrate)

 order=10
 lag=10
 win_length=20

 pre_len = order + win_length + lag
 #### read data of length pre_len
 thres1=0.2 #(normally, thres2 < thres1) thres1 is threshold for detecting anomalies' starts
 thres2=0.1 #thres2 is threshold for detecting anomalies' starts
 state=0
 #maxbuffersize=pre_len
 windowSize=pre_len #elementsNumberHR = #hrTimeWindow * samplingrate

 



 # Buffers for time and
 buffer      = []
 buffertime  = []

#  alg.logpath = ""
# Getting the user input parameters
#  global ip, rip

 unit = sys.argv[1]

 if(len(sys.argv) > 4):
   ip = sys.argv[4] # influxdb IP address
 

 if(len(sys.argv) > 5):
    ssl = str2bool(sys.argv[5]) #https or http
    httpStr = sys.argv[5]+"://"
 else:
    ssl = True
    httpStr = "https://"

 if(len(sys.argv) > 2):
    current = local_time_epoch(sys.argv[2], "America/New_York")
    print(current)
 else:
    current = datetime.now().timestamp()
    bDependOnMovement = True

 if(len(sys.argv) > 3):
     endSet = True
     end = local_time_epoch(sys.argv[3], "America/New_York")

 else:
     endSet = False
     end = datetime.now().timestamp() # never will be used, just give a value to avoid compile errors

 endEpoch = end # int( (end - datetime(1970,1,1)).total_seconds())

# Determining the starting point of the buffer using epoch time
 epoch2 = current  #) # int( (current - datetime(1970,1,1)).total_seconds())
 startEpoch = datetime.fromtimestamp(epoch2).isoformat()
 
 print(startEpoch)

 print("len(sys.argv)", len(sys.argv))
 print("### Current time:", current, " ### \n")
 print("### End time:", end, " ### \n")
#  url = httpStr + rip + ":3000/d/o2RBARGMz/bed-dashboard-algtest?var-mac=" + str(unit)

#  if(len(sys.argv) > 2):
#     url = url + "&from=" + str(int(startEpoch*1000)) #+ "000" 
#  else:
#     url = url + "&from=now-2m"

#  if(len(sys.argv) > 3):
#     url = url + "&to=" + str(int(endEpoch*1000)) #+ "000"
#  else:
#     url = url + "&to=now"
#  name = 'vitalsigns'
#  url = url + f"&var-name={name}&orgId=1&refresh=3s"


#  print("Click here to see the results in Grafana:\n\n" + url)
# #  input("Press any key to continue")
#  webbrowser.open(url, new=2)

 
 #print("client:",ip,port, user, passw, db, ssl)
#  try:
#    client = influxdb_client.InfluxDBClient(
#     url=url,
#     token=token,
#     org=org
# )
#  except Exception as e:
#    print("main(), DB access error:")
#    print("Error", e)
#    quit()


 # set max retries for DB query
 numTry = 0 
 MAXTRY = 100 # max try of 100 seconds
 result = []



 #current = datetime.utcnow().timestamp()
 fs = 10
 # Parameters for the Query
#  epoch2 = epoch2 - pre_len/fs
#  epoch1 = epoch2 - pre_len/fs   #int((pre_len/fs)*1000)
 
 epoch1=epoch2 + pre_len/fs
#  dt_epoch2 = datetime.fromtimestamp(epoch2)
 epoch2_ios = datetime.fromtimestamp(epoch2).isoformat()
 #dt_epoch2 = datetime.fromtimestamp(epoch2)
#  print("dt_epoch1 =", dt_epoch2)
 print("epoch2 =", epoch2)
 #print("dt_epoch2 =", dt_epoch2)

 #print(epoch1,epoch2)

 str=time.time()
#############################  CHANGE TO NEW FORMAT

 startdata, times = read_influx2(src, unit, 'NI_Waveform', 'sensor1_AC_mag', epoch2_ios, pre_len, startEpoch) # sensor2_DC_mag
 print(times)
 end=time.time()
 datatime=end-str
 print("time of reading the data:",datatime)

 startdata = np.array(startdata)

 print("shape of the startdata:", startdata.shape, times)
 print("time length of the window:")
 

#  timein = datetime.strptime(times[pre_len-1],"%Y-%m-%dT%H:%M:%S.%fZ")
#  timeout = datetime.strptime(times[0],"%Y-%m-%dT%H:%M:%S.%fZ")
 timein = times[-1]
 timeout = times[0]

 print(timein-timeout)
 #start=data[:pre_len]  #### get the start data which is required to initiate the algorithm, the length is "pre_len"
 

 Score_start=np.zeros(1) # get the initial score, Score_start
 x1 = np.empty(order, dtype=np.float64) 
 x1 = np.random.rand(order)
 print("shape of x1:",x1.shape)
 x1 /= np.linalg.norm(x1)
 score_start, x1 = SingularSpectrumTransformation(win_length=win_length, x0=x1, n_components=2,order=order, lag=lag,is_scaled=True).score_online(startdata)
 Score_start=score_start+Score_start*10**5


 #Score_start, x1 = detect.start_SST(startdata=startdata,win_length=win_length, n_component=2,order=order, lag=lag)

 print("start score:",Score_start)

#  epoch2 = current
#  epoch1 = epoch1+ pre_len/fs
 #print(epoch1,epoch2)
 # Infinite Loop
 j=0
 while True:
    j=j+1
    print(j)
    fs=10
    # Cheking is the process need to sleep
    #current = datetime.utcnow().timestamp() #(datetime.utcnow() - datetime(1970,1,1)).total_seconds()
    #epoch2 = epoch2 + 2
    # epoch1=epoch1+pre_len/fs
    epoch2 = epoch2 + pre_len/fs
    print(epoch1,epoch2)
   #  epoch2 = epoch2
   #  epoch1 = epoch2 - 1 
    if (endSet == False and (current-epoch2) < 1): 
        time.sleep(1)
        if(debug): print("*********")

#    if(debug): print("*****************************************"+str(statusKey))
    if (endSet and epoch2 > endEpoch):
        if(debug): print("**** Ended as ", epoch2, " > ", end, " ***")
        #print("Click here to see the results in Grafana:\n\n" + url)
        if(len(sys.argv) < 3):
          quit()
    
    print('start:', epoch1, 'end:', epoch2)
    epoch2_ios = datetime.fromtimestamp(epoch2).isoformat()

    try:
        #############################  CHANGE TO NEW FORMAT
        values, times = read_influx2(src, unit, 'NI_Waveform', 'sensor1_AC_mag', epoch2_ios, pre_len, startEpoch)
        print("shape of the data being through",len(values))
    except Exception as e:
        print("main(), no data in the query time period:")
        print("Error", e)
        time.sleep(1)
        numTry += 1
        if (numTry > MAXTRY):            
            quit()
    
    # query = 'SELECT "value" FROM Z WHERE ("location" = \''+unit+'\')  and time >= '+ str(int(epoch1*10e8))+' and time <= '+str(int(epoch2*10e8))
    # print(query)

    # try:
    #     result = client.query(query)
    # except Exception as e:
    #     print("main(), no data in the query time period:")
    #     print("Error", e)
    #     time.sleep(1)
    #     numTry += 1
    #     if (numTry > MAXTRY):            
    #         quit()
    # # print(result)
    # points = list(result.get_points())
    # values =  list(map(operator.itemgetter('value'), points))
    # times  =  list(map(operator.itemgetter('time'),  points))

    #the buffer management modules
    # buffertime = buffertime + times
    # buffer     = buffer + values
    # buffLen    = len(buffer)
    # if(debug): 
    #     print("buffLen: ", buffLen) 
    #     if(buffLen>0):
    #         print("Buffer Time:    " + buffertime[0]+ "  -   " + buffertime[buffLen-1])
    #         # print("Buffer Time:    " + epoch_time_local(buffertime[0], "America/New_York") + "  -   " + epoch_time_local(buffertime[buffLen-1], "America/New_York"))
    # #  Cutting the buffer when overflow
    # if(buffLen > maxbuffersize):
    #    difSize = buffLen - maxbuffersize
    #    del buffer[0:difSize]
    #    del buffertime[0:difSize]
    #    buffLen    = buffLen - difSize
    # # get more data if the buffer does not have enough data for the minimum window size
    # # if (buffLen < windowSize):
    # #     continue
    
    # data = buffer[buffLen-windowSize:buffLen]
    # nowtime = buffertime[buffLen-1]
    data=values
    #  the blood pressure estimation algorithm
    if(debug): print("Calculating vital signs")

    stream=np.array(data)  #### the new data coming through
    print("Shape of stream data: ",stream.shape)
    # lastdata=start ### the initial start of the algorithm
    score,duration,x1=stream_SST(stream,win_length,n_component=2,order=order,lag=lag,x0=x1) #,state_last=state,thres1=thres1,thres2=thres2
    print("score of this window:", score)

    if score >= thres1:  #and state_last==0  
      print("the anomaly starts") 
      state=1 
    else:
      state=0

    print("state of this window is :", state)

    #print('nowtime:', nowtime)
    print("The anomaly score for current time point is ",score)
    print("The time that processes", duration)
    print("The current state is:", state)

    #hr,rr,bph,bpl = alg.predict(data, fs=100, cutoff=4,nlags=200,order=1)
    #if(debug): print('hr:', hr, ' rr:', rr, ' bph:', bph, ' bpl:', bpl)
    

    timestamp = int(epoch2* 1000000000)   #locacl_time_epoch(str(nowtime[:-1]), "UTC")
    #print(epoch1)
    #print(nowtime[:-1])
    # dt_write = datetime.fromtimestamp(epoch1)
    # print("dt_epoch1 =", dt_write)

    #############################  CHANGE TO NEW FORMAT
    write_influx2(dest, unit, 'sensor1_AC_mag_score', 'score', [score], timestamp, 1)
    write_influx2(dest, unit, 'sensor1_AC_mag_score', 'state', [state], timestamp, 1)
    # tz_NY = pytz.timezone("America/New_York")
    # currentTime = datetime.now(tz_NY)
    # timestamp = int(currentTime.timestamp()* 1000000000)
    # print(currentTime.timestamp())
    # dt_write = datetime.fromtimestamp(currentTime.timestamp() )
    # print("dt_epoch1 =", dt_write)

    #############################  CHANGE TO NEW FORMAT
    # writeData = [
    #     {
    #         "measurement": "sensor2_ph1_mag_score",
    #         "tags": {"location": unit},
    #         "fields": {
    #             "score": score,
    #             "state": state,
    #         },
    #         "time": timestamp,
    #     }
    # ]
    # print(unit)
    #############################  CHANGE TO NEW FORMAT
    # client.write_points(
    #     writeData, time_precision="n", batch_size=1, protocol="json"  #### check the writing limit: if there is anything
    # )

    #write_influx(dest, unit, 'sensor1_DC_score', 'score', [score], timestamp, 1)
    #write_influx(dest, unit, 'sensor1_DC_state', 'state', [state], timestamp, 1)
   #  write_influx(dest, unit, 'bpressure', 'bph', bph, timestamp, fs)
   #  write_influx(dest, unit, 'bpressure', 'bpl', bpl, timestamp, fs)
    # end of adding

if __name__== '__main__':
  main()