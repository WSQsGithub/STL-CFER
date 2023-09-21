import numpy as np
import json
import pandas as pd

# from IPython.display import display
from tabulate import tabulate
import random
import copy
from scipy.io import savemat, loadmat

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

from datetime import datetime
import time
import sys
import math
import itertools

from tqdm import tqdm
# from IPython.display import display, clear_output
# import plotly.graph_objects as go
# from plotly.offline import iplot
import os

from torch.utils.tensorboard import SummaryWriter

def packState(state, f_state) -> str:
    '''
    pack numpy state into string
    '''
    packed_state = np.concatenate((state, f_state))
    return str(packed_state)

# def unpackState(state_str) -> list:
#     '''
#     turn string to array (state, f_state)
#     '''
#     return state_str.split()

def readLog(file_path):
    '''
    read .Mat file and return data for analysis
    '''
    mat = loadmat(file_path)
    
    return mat["train_log"], mat["test_log"]

def getSampleTrace(test_log):
    '''
    return a trace with the most reward as the sample path
    ''' 
    traces = test_log["state"]
    rewards = test_log["episode_reward"] # no idea why it has so many wrappings...
    sample_id = np.argmax(rewards)
    sample_trace = traces[sample_id,:,:]
    sample_trace_reward = test_log['reward'][sample_id,:,]
    return sample_trace, sample_trace_reward

def plotSampleTrace(world, test_log):
    sample_trace, sample_trace_reward = getSampleTrace(test_log)
    world.showWorld(sample_trace)
    
    fig, axs = plt.subplots(3,1, figsize=(5,5))
    
    # subplot 1：X-t plot
    axs[0].plot(range(len(sample_trace[:, 0])), sample_trace[:, 0], '-o',linewidth=2)
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('X')

    # subplot 2：Y-t plot
    axs[1].plot(range(len(sample_trace[:, 1])), sample_trace[:, 1], '-o',linewidth=2)
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('Y')

    # subplot 3：R-t plot
    axs[2].plot(range(len(sample_trace_reward)), sample_trace_reward, '-o',linewidth=2)
    axs[2].set_xlabel('t')
    axs[2].set_ylabel('R')
    
    # adjust the layout
    plt.tight_layout()
    
    plt.show(block=False)
    
    plt.pause(1)
    
def getLearningCurveData(train_log):
    
    return train_log["sat"]

def smooth_array(arr, window_size):
    window = np.ones(window_size) / window_size
    smoothed_arr = np.convolve(arr, window, mode='valid')
    return smoothed_arr


def smoothData(arr, window_size):
    # pad the missing signal
    padding = np.full(window_size-1, 0)
    padded_arr = np.concatenate([padding, arr])
    
    rolling_mean = smooth_array(padded_arr, window_size)
    rolling_std = np.std([padded_arr[i:i+window_size] for i in range(len(padded_arr)-window_size+1)], axis=1)
    
    lower_bounds = [avg - std for avg, std in zip(rolling_mean, rolling_std)]
    upper_bounds = [avg + std for avg, std in zip(rolling_mean, rolling_std)]
    
    return rolling_mean, lower_bounds, upper_bounds

def curveData(arr, window_size):
    rolling_mean = np.average([arr[i*window_size:(i+1)*window_size] for i in range(int(round(len(arr)/window_size))-1)],axis=1)
    rolling_std = np.std([arr[i*window_size:(i+1)*window_size] for i in range(int(round(len(arr)/window_size))-1)],axis=1)
    lower_bounds = [avg - std for avg, std in zip(rolling_mean, rolling_std)]
    upper_bounds = [avg + std for avg, std in zip(rolling_mean, rolling_std)]
    return rolling_mean, lower_bounds, upper_bounds
