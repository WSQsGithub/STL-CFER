import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from core.world import *

plt.rcParams['pdf.fonttype']=42



# why not add 4 file on the same page?
# requirement: 
# 4 btn to read files for naive, naiveER, CFER and CFPER, clear path if click again
# 4 btn to plot the best trace or a random trace in a toggle mode
# 1 btn to draw the sucess rate curve
# 1 btn to draw the reward curve
# a text label that show which file has been selected and the p_sat in each file
class MatplotlibApp:
    def __init__(self, root):

        self.grid_size = 5 # should depend on the file I choose, but now I'll settle with code changing

        obstacle = np.array([[0,0.2,0.2,0.2],[0,0.4,0.4,0.2],[0.4,0,0.2,0.2],[0.6,0.4,0.4,0.2],[0.4,0.8,0.2,0.2]])
        # obstacle = None
        region = np.array([[0.6,0,0.4,0.4],[0,0.6,0.4,0.4],[0.6,0.6,0.4,0.4]])

        self.world = World(grid_size=self.grid_size, 
                    region=region,  
                    prob_right=0.91, 
                    beta=50,scale=[1,0.01],obstacle=obstacle)
    
        self.root = root
        self.root.title("Matplotlib Plotter")
        self.file_path = {
            "naive" : "",
            "naiveER" : "",
            "CFER": "",
            "CFPER": ""
        }
        self.window_size = 20
        self.colors = {"naive":"#C848b9",
                       "naiveER":"#F962A7",
                       "CFER":"#FD836D",
                       "CFPER":"#FFBA69"}
        self.createGridsizeButton()
        self.createChooseFileButton()
        self.createPlotTraceButton()
        self.createPlotCurveButton()
        self.trace_mode = "best"

    def createGridsizeButton(self):
        self.grid_btn = tk.Button(self.root, text="grid_5", command=self.toggleGrid)
        self.grid_btn.grid(row = 4, column = 3, padx = 10, pady = 10)

    def toggleGrid(self):
        if self.grid_size == 5:
            self.grid_size = 10
        else:
            self.grid_size = 5
        self.world.updateSize(self.grid_size)
        self.grid_btn.config(text=f"grid_{self.grid_size}")

    
    def createChooseFileButton(self):
        self.choose_btn = dict()
        index = 0
        for alg in self.file_path.keys():
            self.choose_btn[alg] = tk.Button(self.root, text=alg, command=lambda alg_name=alg: self.choose_file(alg_name))
            self.choose_btn[alg].grid(row = 0, column = index, padx = 10, pady = 10)
            index += 1
            
            
    def createPlotTraceButton(self):
        self.trace_btn = dict()
        index = 0
        for alg in self.file_path.keys():
            print(alg)
            self.trace_btn[alg] = tk.Button(self.root, text="plot")
            self.trace_btn[alg].config(command=lambda alg_name=alg: self.plot_trace(alg_name))
            self.trace_btn[alg].grid(row = 1, column = index, padx = 10, pady = 10)
            index += 1
            
    def createPlotCurveButton(self):
        self.curve_btn_Win = tk.Button(self.root, text="Sat_Win", command=lambda mode="Win": self.plot_curve(mode))
        self.curve_btn_Seg = tk.Button(self.root, text="Sat_Seg", command=lambda mode="Seg": self.plot_curve(mode))

        self.reward_btn_Win = tk.Button(self.root, text="R_Win", command=lambda mode="Win": self.plot_reward(mode))
        self.reward_btn_Seg = tk.Button(self.root, text="R_Seg", command=lambda mode="Seg": self.plot_reward(mode))

        self.curve_btn_Seg.grid(row = 2, column = 0, padx = 10, pady = 10)
        self.curve_btn_Win.grid(row = 2, column = 1, padx = 10, pady = 10)
        self.reward_btn_Seg.grid(row = 2, column = 2, padx = 10, pady = 10)
        self.reward_btn_Win.grid(row = 2, column = 3, padx = 10, pady = 10)
        
        

    def plot_reward(self, mode):
        non_empty_keys_list = get_non_empty_keys(self.file_path)
        if non_empty_keys_list:
            print("Non-empty keys:", non_empty_keys_list)
            fig, ax = plt.subplots(figsize=(5,4))
            for alg in non_empty_keys_list:
                filepath = self.file_path[alg]
                train_log, _ = readLog(filepath)
                data = getLearningRewardData(train_log)
                if mode == "Seg":
                    smooth_arr, lower_bounds, upper_bounds = curveData(data, self.window_size)
                    x = np.array(range(len(smooth_arr)))*self.window_size
                    
                    plt.plot(x, smooth_arr, c=self.colors[alg],linewidth=2, label=alg)    
                    plt.fill_between(x, lower_bounds, upper_bounds, facecolor=self.colors[alg], alpha=0.2)
                else:
                    smooth_arr, lower_bounds, upper_bounds = smoothData(data, self.window_size)
                    x = np.array(range(len(smooth_arr)))
                    plt.plot(x[self.window_size:-1], smooth_arr[self.window_size:-1], c=self.colors[alg],linewidth=2, label=alg)    
                    plt.fill_between(x[self.window_size:-1], lower_bounds[self.window_size:-1], upper_bounds[self.window_size:-1], facecolor=self.colors[alg], alpha=0.2)
                
            plt.legend(loc='lower right')
            plt.show(block=False)
                
        else:
            print("All dictionary values are empty.")



    def plot_curve(self, mode):
        non_empty_keys_list = get_non_empty_keys(self.file_path)
        if non_empty_keys_list:
            print("Non-empty keys:", non_empty_keys_list)
            fig, ax = plt.subplots(figsize=(5,4))
            for alg in non_empty_keys_list:
                filepath = self.file_path[alg]
                train_log, _ = readLog(filepath)
                data = getLearningCurveData(train_log)
                if mode == "Seg":
                    smooth_arr, lower_bounds, upper_bounds = curveData(data, self.window_size)
                    x = np.array(range(len(smooth_arr)))*self.window_size
                else:
                    smooth_arr, lower_bounds, upper_bounds = smoothData(data, self.window_size)
                    x = np.array(range(len(smooth_arr)))
                    
                
                plt.plot(x, smooth_arr, c=self.colors[alg],linewidth=2, label=alg)    
                plt.fill_between(x, lower_bounds, upper_bounds, facecolor=self.colors[alg], alpha=0.2)
                
            plt.legend(loc='lower right')
            plt.show(block=False)
                
        else:
            print("All dictionary values are empty.")

            
    def choose_file(self, alg_name):
        if self.file_path[alg_name] == "":
            file_path = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")], initialdir='../data')
            if file_path:
                self.file_path[alg_name]  = file_path
            self.choose_btn[alg_name].config(text="clear")
        else:
            self.choose_btn[alg_name].config(text=alg_name)
            self.file_path[alg_name] = ""
            
           
        if self.file_path:
            print(">>> Selected files:", self.file_path)
        
    def plot_trace(self, alg_name):
        path = self.file_path[alg_name]
        if path == "":
            print(">>> Please select file first!")
            return
        else:
            train_log, test_log = readLog(path)
            print(">>> P_sat = ", test_log["p_sat"][0][0][0][0])
            plotSampleTrace(self.world, test_log, self.trace_mode)
            if self.trace_mode == "best":
                self.trace_mode = "random"
            else:
                # self.trace_btn[alg_name].config(text="best")
                self.trace_mode = "best"
            print(self.trace_mode)
                    
                    
    
def readLog(file_path):
    '''
    read .Mat file and return data for analysis
    '''
    mat = loadmat(file_path)
    
    return mat["train_log"], mat["test_log"]

def getSampleTrace(test_log : object, type="best"):
    '''
    return a trace with the most reward as the sample path
    ''' 
    traces = test_log["state"][0][0]
    rewards = test_log["episode_reward"][0][0][0]
    if type == "best":
        sample_id = np.argmax(rewards)
    elif type == "random":
        sample_id = np.random.randint(low=0, high=len(rewards))
    print(sample_id)
    sample_trace = traces[:,:,sample_id]
    sample_trace_reward = test_log['reward'][0][0][:,sample_id]
    return sample_trace, sample_trace_reward

def plotSampleTrace(world, test_log, type="best"):
    print(world.grid_size)
    sample_trace, sample_trace_reward = getSampleTrace(test_log, type)
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
    
    
def get_non_empty_keys(dictionary):
    non_empty_keys = []
    for key, value in dictionary.items():
        if value != "":
            non_empty_keys.append(key)
    return non_empty_keys
            
            
# Win
def curveData(arr, window_size):
    rolling_mean = np.average([arr[i*window_size:(i+1)*window_size] for i in range(int(round(len(arr)/window_size))-1)],axis=1)
    rolling_std = np.std([arr[i*window_size:(i+1)*window_size] for i in range(int(round(len(arr)/window_size))-1)],axis=1)
    lower_bounds = [avg - std for avg, std in zip(rolling_mean, rolling_std)]
    upper_bounds = [avg + std for avg, std in zip(rolling_mean, rolling_std)]
    return rolling_mean, lower_bounds, upper_bounds
            
            
# Seg
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
            
            
def getLearningCurveData(train_log):
    
    return train_log["sat"][0][0][0]

def getLearningRewardData(train_log):
    
    return train_log["episode_reward"][0][0][0]

            
root = tk.Tk()
app = MatplotlibApp(root)
root.mainloop()
