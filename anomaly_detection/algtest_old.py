'''
Author: Qi7
Date: 2023-03-02 11:10:23
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-03-02 23:08:33
Description: online anomly detection with SST
'''
from util import *
from fastsst.sst import *
from datetime import datetime
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
import time
import sys, os
import warnings
import numpy as np

warnings.filterwarnings("ignore")

#influxdb config
token = "0ML4vBa-81dGKI3_wD-ReiSRdLggdJPXKoTKLPITBcOZXl8MJh7W8wFSkNUNM_uPS9mJpzvBxUKfKgie0dHiow=="
org = "lab711"
bucket = "testbed"
url = "http://sensorwebdata.engr.uga.edu:8086"
measurement = "detection_results"

debug = True
verbose = True

src = {'ip': url, 'org':org,'token':token,'bucket':bucket}
dest = {'ip': url, 'org':org,'token':token,'bucket':bucket}

def main():
    progname = sys.argv[0]
    if(len(sys.argv)<2):
        print(f"Usage: {progname} <location>")
        print(f"Example: {progname} lab711")
    
    order=10
    lag=10
    win_length=20
    pre_len = order + win_length + lag # length of data been analyzed
    #### read data of length pre_len
    thres1=0.4 #(normally, thres2 < thres1) thres1 is threshold for detecting anomalies' starts
    thres2=0.1 #thres2 is threshold for detecting anomalies' starts
    state=0

    
    location = sys.argv[1]
    current = datetime.now().timestamp()
    end = datetime.now().timestamp()
    endSet = False
    
    endEpoch = end
    epoch2 = current
    startEpoch = datetime.fromtimestamp(epoch2).isoformat()
    
    numTry = 0 
    MAXTRY = 100 # max try of 100 seconds
    fs = 10
    epoch1 = epoch2 + pre_len/fs
    epoch2_ios = datetime.fromtimestamp(epoch2).isoformat()
    
    startdata, times = read_influx2(src, location, 'NI_Waveform', 'sensor1_AC_mag', epoch2_ios, pre_len, startEpoch)
    
    startdata = np.array(startdata)
    print("shape of the startdata:", startdata.shape, times)
    print(f"time length of the window: {times[-1] - times[0]}")
    
    score_start = np.zeros(1) # get the initial score
    
    x1 = np.random.rand(order)
    
    score, x1 = SingularSpectrumTransformation(win_length=win_length, x0=x1, n_components=2,order=order, lag=lag,is_scaled=True).score_online(startdata)
    score_start = score + score_start * 10 ** 5
    
    print(f"start score: {score_start}")
    
    # infinite loop
    j = 0
    while True:
        j = j + 1
        epoch2 += pre_len / fs
        if verbose: print(f"epoch1: {epoch1}; epoch2: {epoch2}")
        if (endSet == False and (current - epoch2) < 1):
            times.sleep(1)
        
        if (endSet and epoch2 > endEpoch):
            if(debug): print("**** Ended as ", epoch2, " > ", end, " ***")
            if(len(sys.argv) < 3):
                quit()
        
        print('start:', epoch1, 'end:', epoch2)
        epoch2_ios = datetime.fromtimestamp(epoch2).isoformat()
        
        try:
        #############################  CHANGE TO NEW FORMAT
            values, times = read_influx2(src, location, 'NI_Waveform', 'sensor1_AC_mag', epoch2_ios, pre_len, startEpoch)
            if verbose: print(f"length of the data being through :{len(values)}")
        except Exception as e:
            print("main(), no data in the query time period:")
            print("Error", e)
            time.sleep(1)
            numTry += 1
            if (numTry > MAXTRY):            
                quit()
        
        