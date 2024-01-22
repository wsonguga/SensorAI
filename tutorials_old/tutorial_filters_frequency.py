import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from pathlib import Path

import streamlit as st

import re
import pytz
from datetime import datetime

import enum
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.metrics import classification_report, auc, roc_curve, roc_auc_score

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

import joblib

import tutorials_old.load_data as load_data
import sk_classifier_builder as skb
import tutorials_old.sk_classifier_metrics as skm
import filter_builder as fltr

import tutorials_old.load_data as ld

st.title('Filter Demo')

p = Path('.')

save_eval_path = p / "AI_engine/evaluation_results/"
save_model_path = p / "AI_engine/saved_models/"
datapath = p / "AI_engine/test_data/"


def filter_plot(filter_name,sig,t,fs,lowcut,highcut,orders=[5],max_rip=5,min_attn=5):
  import numpy as np
  import matplotlib.pyplot as plt
  from scipy.signal import freqz

  # Plot the frequency response for a few different orders.
  plt.figure(1)
  plt.clf()
  for order in orders:
    if filter_name == 'butter_bp':
      b, a = fltr.butter_bandpass(lowcut, highcut, fs, order=order)
    elif filter_name == 'butter_hp':
      b, a = fltr.butter_highpass(lowcut, fs, order=order)
    elif filter_name == 'butter_lp':
      b, a = fltr.butter_lowpass(highcut, fs, order=order)
    elif filter_name == 'cheby1_bp':
      b, a = fltr.cheby1_bandpass(max_rip=max_rip, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    elif filter_name == 'cheby1_hp':
      b, a = fltr.cheby1_highpass(max_rip, lowcut, fs, order=order)
    elif filter_name == 'cheby1_lp':
      b, a = fltr.cheby1_lowpass(max_rip, highcut, fs, order=order)
    elif filter_name == 'cheby2_bp':
      b, a = fltr.cheby2_bandpass(min_attn=min_attn, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    elif filter_name == 'cheby2_hp':
      b, a = fltr.cheby2_highpass(min_attn, lowcut, fs, order=order)
    elif filter_name == 'cheby2_lp':
      b, a = fltr.cheby2_lowpass(min_attn, highcut, fs, order=order)
    elif filter_name == 'butter_bs':
      b, a = fltr.butter_bandstop(lowcut, highcut, fs, order=order)
    elif filter_name == 'cheby1_bs':
      b, a = fltr.cheby1_bandstop(max_rip=max_rip, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    elif filter_name == 'cheby2_bs':
      b, a = fltr.cheby2_bandstop(min_attn=min_attn, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    elif filter_name == 'bessel_bp':
      b, a = fltr.bessel_bandpass(lowcut, highcut, fs, order=order)
    elif filter_name == 'bessel_hp':
      b, a = fltr.bessel_highpass(lowcut, fs, order=order)
    elif filter_name == 'bessel_lp':
      b, a = fltr.bessel_lowpass(highcut, fs, order=order)
    elif filter_name == 'bessel_bs':
      b, a = fltr.bessel_bandstop(lowcut, highcut, fs, order=order)
    else:
      b, a = fltr.butter_bandpass(lowcut, highcut, fs, order=order)
    w, h = freqz(b, a, worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

  plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],'--', label='sqrt(0.5)')
  plt.title(filter_name + " frequency response")
  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Gain')
  plt.grid(True)
  plt.legend(loc='best')

  # Filter a noisy signal.
  #f0 = 20.0
  x = sig
  plt.figure(2)
  plt.clf()
  plt.plot(t, x, label='Noisy signal')

  # Butterworth filters
  if filter_name == 'butter_bp':
    f0 = int((highcut-lowcut)/2+lowcut)
    label='Filtered signal (%g Hz)' % f0
    y = fltr.butter_bandpass_filter(x, lowcut, highcut, fs, order=orders[-1])
  elif filter_name == 'butter_bs':
    f0 = int((highcut-lowcut)/2+lowcut)
    label='Blocked signal (%g Hz)' % f0
    y = fltr.butter_bandstop_filter(x, lowcut, highcut, fs, order=orders[-1])
  elif filter_name == 'butter_hp':
    f0 = int(lowcut)
    label='Signals above (%g Hz)' % f0
    y = fltr.butter_highpass_filter(x, lowcut, fs, order=orders[-1])
  elif filter_name == 'butter_lp':
    f0 = int(highcut)
    label='Signals below (%g Hz)' % f0
    y = fltr.butter_lowpass_filter(x, highcut, fs, order=orders[-1])
  # Cheby1 filters
  elif filter_name == 'cheby1_bp':
    f0 = int((highcut-lowcut)/2+lowcut)
    label='Filtered signal (%g Hz)' % f0
    y = fltr.cheby1_bandpass_filter(x, max_rip,lowcut, highcut, fs, order=orders[-1])
  elif filter_name == 'cheby1_bs':
    f0 = int((highcut-lowcut)/2+lowcut)
    label='Blocked signal (%g Hz)' % f0
    y = fltr.cheby1_bandstop_filter(x, max_rip, lowcut, highcut, fs, order=orders[-1])
  elif filter_name == 'cheby1_hp':
    f0 = int(lowcut)
    label='Signals above (%g Hz)' % f0
    y = fltr.cheby1_highpass_filter(x, max_rip,lowcut, fs, order=orders[-1])
  elif filter_name == 'cheby1_lp':
    f0 = int(highcut)
    label='Signals below (%g Hz)' % f0
    y = fltr.cheby1_lowpass_filter(x, max_rip, highcut, fs, order=orders[-1])
  # Cheby2 filters
  elif filter_name == 'cheby2_bp':
    f0 = int((highcut-lowcut)/2+lowcut)
    label='Filtered signal (%g Hz)' % f0
    y = fltr.cheby2_bandpass_filter(x, min_attn, lowcut, highcut, fs, order=orders[-1])
  elif filter_name == 'cheby2_bs':
    f0 = int((highcut-lowcut)/2+lowcut)
    label='Blocked signal (%g Hz)' % f0
    y = fltr.cheby2_bandstop_filter(x, min_attn, lowcut, highcut, fs, order=orders[-1])
  elif filter_name == 'cheby2_hp':
    f0 = int(lowcut)
    label='Signals above (%g Hz)' % f0
    y = fltr.cheby2_highpass_filter(x, min_attn, lowcut, fs, order=orders[-1])
  elif filter_name == 'cheby2_lp':
    f0 = int(highcut)
    label='Signals below (%g Hz)' % f0
    y = fltr.cheby2_lowpass_filter(x, min_attn, highcut, fs, order=orders[-1])
  # Bessel filters
  if filter_name == 'bessel_bp':
    f0 = int((highcut-lowcut)/2+lowcut)
    label='Filtered signal (%g Hz)' % f0
    y = fltr.bessel_bandpass_filter(x, lowcut, highcut, fs, order=orders[-1])
  elif filter_name == 'bessel_bs':
    f0 = int((highcut-lowcut)/2+lowcut)
    label='Blocked signal (%g Hz)' % f0
    y = fltr.bessel_bandstop_filter(x, lowcut, highcut, fs, order=orders[-1])
  elif filter_name == 'bessel_hp':
    f0 = int(lowcut)
    label='Signals above (%g Hz)' % f0
    y = fltr.bessel_highpass_filter(x, lowcut, fs, order=orders[-1])
  elif filter_name == 'bessel_lp':
    f0 = int(highcut)
    label='Signals below (%g Hz)' % f0
    y = fltr.bessel_lowpass_filter(x, highcut, fs, order=orders[-1])
  # Elliptic filters
  elif filter_name == 'ellip_bp':
    f0 = int((highcut-lowcut)/2+lowcut)
    label='Filtered signal (%g Hz)' % f0
    y = fltr.ellip_bandpass_filter(x, max_rip, min_attn, lowcut, highcut, fs, order=orders[-1])
  elif filter_name == 'ellip_bs':
    f0 = int((highcut-lowcut)/2+lowcut)
    label='Blocked signal (%g Hz)' % f0
    y = fltr.ellip_bandstop_filter(x, max_rip, min_attn, lowcut, highcut, fs, order=orders[-1])
  elif filter_name == 'ellip_hp':
    f0 = int(lowcut)
    label='Signals above (%g Hz)' % f0
    y = fltr.ellip_highpass_filter(x, max_rip, min_attn, lowcut, fs, order=orders[-1])
  elif filter_name == 'ellip_lp':
    f0 = int(highcut)
    label='Signals below (%g Hz)' % f0
    y = fltr.ellip_lowpass_filter(x, max_rip, min_attn, highcut, fs, order=orders[-1])
  else:
    f0 = int((highcut-lowcut)/2+lowcut)
    label='Filtered signal (%g Hz)' % f0
    y = fltr.butter_bandpass_filter(x, lowcut, highcut, fs, order=orders[-1])
  plt.plot(t, y, label=label)
  plt.title(filter_name + " filtered signal")
  plt.xlabel('time (seconds)')
  plt.grid(True)
  plt.axis('tight')
  plt.legend(loc='upper left')

  plt.show()


# Sample rate and desired cutoff frequencies (in Hz).
fs = 60.0
lowcut = 15.0
highcut = 25.0
orders = [3,6,9]

# Filter a noisy signal.
T = 1
nsamples = 100 #int(T * fs)
t = np.linspace(0, T, nsamples, endpoint=False)
sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t) + np.sin(2*np.pi*30*t) # 10 Hz + 20 Hz + 30 Hz

#High Pass (butter)
filter_plot('butter_hp',sig,t,fs,lowcut,highcut,orders)

# High Pass (cheby1)
filter_plot('cheby1_hp',sig,t,fs,lowcut,highcut,orders)

# High Pass (cheby2)
filter_plot('cheby2_hp',sig,t,fs,lowcut,highcut,orders)

#High Pass (bessel)
filter_plot('bessel_hp',sig,t,fs,lowcut,highcut,orders)

#High Pass (elliptic)
filter_plot('ellip_hp',sig,t,fs,lowcut,highcut,orders)

#Low Pass (butter)
filter_plot('butter_lp',sig,t,fs,lowcut,highcut,orders)

# Low Pass (cheby1)
filter_plot('cheby1_lp',sig,t,fs,lowcut,highcut,orders)

# Low Pass (cheby2)
filter_plot('cheby2_lp',sig,t,fs,lowcut,highcut,orders)

#Low Pass (bessel)
filter_plot('bessel_lp',sig,t,fs,lowcut,highcut,orders)

#Low Pass (elliptic)
filter_plot('ellip_lp',sig,t,fs,lowcut,highcut,orders)

#Band Pass (butter)
filter_plot('butter_bp',sig,t,fs,lowcut,highcut,orders)

# Band Pass (cheby1)
filter_plot('cheby1_bp',sig,t,fs,lowcut,highcut,orders)

# Band Pass (cheby2)
filter_plot('cheby2_bp',sig,t,fs,lowcut,highcut,orders)

#Band Pass (bessel)
filter_plot('bessel_bp',sig,t,fs,lowcut,highcut,orders)

#Band Pass (elliptic)
filter_plot('ellip_bp',sig,t,fs,lowcut,highcut,orders)

#Band Stop (butter)
filter_plot('butter_bs',sig,t,fs,lowcut,highcut,orders)

# Band Stop (cheby1)
filter_plot('cheby1_bs',sig,t,fs,lowcut,highcut,orders)

# Band Stop (cheby2)
filter_plot('cheby2_bs',sig,t,fs,lowcut,highcut,orders)

#Band Stop (bessel)
filter_plot('bessel_bs',sig,t,fs,lowcut,highcut,orders)

#Band Stop (elliptic)
filter_plot('ellip_bs',sig,t,fs,lowcut,highcut,orders)