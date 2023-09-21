import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype']=42
import numpy as np  
from scipy.io import loadmat


file_path = [
    {
    "naive": "patrolling/naive_tau5_grid5_0905.mat",
    "naiveER":"patrolling/naiveER_tau5_grid5_0905.mat",
    "CFER": "patrolling/CFER_tau5_grid5_0905.mat",
    "CFPER": "patrolling/CFPER_tau5_grid5_0905.mat"
    },
    {
    "naive": "patrolling/naive_tau10_grid5_0905.mat",
    "naiveER":"patrolling/naiveER_tau10_grid5_0905.mat",
    "CFER": "patrolling/CFER_tau10_grid5_0905.mat",
    "CFPER": "patrolling/CFPER_tau10_grid5_0905.mat"
    },
    {
    "naive": "patrolling/naive_tau15_grid5_0905a.mat",
    "naiveER":"patrolling/naiveER_tau15_grid5_0905a.mat",
    "CFER": "patrolling/CFER_tau15_grid5_0905a.mat",
    "CFPER": "patrolling/CFPER_tau15_grid5_0905a.mat"
    },
    {
    "naive": "patrolling/naive_tau10_grid10_0905a.mat",
    "naiveER":"patrolling/naiveER_tau10_grid10_0905a.mat",
    "CFER": "patrolling/CFER_tau10_grid10_0905a.mat",
    "CFPER": "patrolling/CFPER_tau10_grid5_0905a.mat"
    },
    {
    "naive": "patrolling/naive_tau15_grid10_0905a.mat",
    "naiveER":"patrolling/naiveER_tau15_grid10_0905a.mat",
    "CFER": "patrolling/CFER_tau15_grid10_0905a.mat",
    "CFPER": "patrolling/CFPER_tau15_grid10_0905a.mat"
    },
    {
    "naive": "patrolling/naive_tau20_grid10_0905a.mat",
    "naiveER":"patrolling/naiveER_tau20_grid10_0905a.mat",
    "CFER": "patrolling/CFER_tau20_grid10_0905a.mat",
    "CFPER": "patrolling/CFPER_tau20_grid10_0905a.mat"
    },
    
    
    {
    "naive": "reaching/naive_tau5_grid5_0905.mat",
    "naiveER":"reaching/naiveER_tau5_grid5_0905.mat",
    "CFER": "reaching/CFER_tau5_grid5_0905.mat",
    "CFPER": "reaching/CFPER_tau5_grid5_0905.mat"
    },
    {
    "naive": "reaching/naive_tau10_grid5_0905.mat",
    "naiveER":"reaching/naiveER_tau10_grid5_0905.mat",
    "CFER": "reaching/CFER_tau10_grid5_0905.mat",
    "CFPER": "reaching/CFPER_tau10_grid5_0905.mat"
    },
    {
    "naive": "reaching/naive_tau20_grid5_0905.mat",
    "naiveER":"reaching/naiveER_tau20_grid5_0905.mat",
    "CFER": "reaching/CFER_tau20_grid5_0905.mat",
    "CFPER": "reaching/CFPER_tau20_grid5_0905.mat"
    },
    {
    "naive": "reaching/naive_tau5_grid10_0905.mat",
    "naiveER":"reaching/naiveER_tau5_grid10_0905.mat",
    "CFER": "reaching/CFER_tau5_grid10_0905.mat",
    "CFPER": "reaching/CFPER_tau5_grid5_0905.mat"
    },
    {
    "naive": "reaching/naive_tau10_grid10_0905.mat",
    "naiveER":"reaching/naiveER_tau10_grid10_0905.mat",
    "CFER": "reaching/CFER_tau10_grid10_0905.mat",
    "CFPER": "reaching/CFPER_tau10_grid10_0905.mat"
    },
    {
    "naive": "reaching/naive_tau20_grid10_0905.mat",
    "naiveER":"reaching/naiveER_tau20_grid10_0905.mat",
    "CFER": "reaching/CFER_tau20_grid10_0905.mat",
    "CFPER": "reaching/CFPER_tau20_grid10_0905.mat"
    },
    
    {
    "naive": "charging/naive_tau5_grid5_0905.mat",
    "naiveER":"charging/naiveER_tau5_grid5_0905.mat",
    "CFER": "charging/CFER_tau5_grid5_0905.mat",
    "CFPER": "charging/CFPER_tau5_grid5_0905.mat"
    },
    {
    "naive": "charging/naive_tau10_grid5_0905.mat",
    "naiveER":"charging/naiveER_tau10_grid5_0905.mat",
    "CFER": "charging/CFER_tau10_grid5_0905.mat",
    "CFPER": "charging/CFPER_tau10_grid5_0905.mat"
    },
    {
    "naive": "charging/naive_tau20_grid5_0905.mat",
    "naiveER":"charging/naiveER_tau20_grid5_0905.mat",
    "CFER": "charging/CFER_tau20_grid5_0905.mat",
    "CFPER": "charging/CFPER_tau20_grid5_0905.mat"
    },
    {
    "naive": "charging/naive_tau5_grid10_0905.mat",
    "naiveER":"charging/naiveER_tau5_grid10_0905.mat",
    "CFER": "charging/CFER_tau5_grid10_0905.mat",
    "CFPER": "charging/CFPER_tau5_grid5_0905.mat"
    },
    {
    "naive": "charging/naive_tau10_grid10_0905.mat",
    "naiveER":"charging/naiveER_tau10_grid10_0905.mat",
    "CFER": "charging/CFER_tau10_grid10_0905.mat",
    "CFPER": "charging/CFPER_tau10_grid10_0905.mat"
    },
    {
    "naive": "charging/naive_tau20_grid10_0905.mat",
    "naiveER":"charging/naiveER_tau20_grid10_0905.mat",
    "CFER": "charging/CFER_tau20_grid10_0905.mat",
    "CFPER": "charging/CFPER_tau20_grid10_0905.mat"
    },
    
    

]




file_name = [
    'patrolling_tau5_grid5.pdf',
    'patrolling_tau10_grid5.pdf',
    'patrolling_tau15_grid5.pdf',
    'patrolling_tau10_grid10.pdf',
    'patrolling_tau15_grid10.pdf',
    'patrolling_tau20_grid10.pdf',
    'reaching_tau5_grid5.pdf',
    'reaching_tau10_grid5.pdf',
    'reaching_tau20_grid5.pdf',
    'reaching_tau5_grid10.pdf',
    'reaching_tau10_grid10.pdf',
    'reaching_tau20_grid10.pdf',
    'charging_tau5_grid5.pdf',
    'charging_tau10_grid5.pdf',
    'charging_tau20_grid5.pdf',
    'charging_tau5_grid10.pdf',
    'charging_tau10_grid10.pdf',
    'charging_tau20_grid10.pdf'
             ]


class App():
    def __init__(self,file_path, file_name, file_dir) -> None:
        self.window_size = 50
        
        self.file_name = file_name
        self.file_dir  = file_dir
        self.file_path = file_path
        
        self.curve_file_dir = '../images/reward/'
        self.sat_file_dir = '../images/sat/'

        self.keylist = ['naive', 'naiveER', 'CFER']

        self.colors = {"naive":"#C848b9",
                "naiveER":"#F962A7",
                "CFER":"#FD836D",
                "CFPER":"#FFBA69"}

    def plot_reward(self, mode, save=False):
        for item in self.file_path:
            if self.keylist:
                print("Non-empty keys:", self.keylist)
                fig, ax = plt.subplots(figsize=(5,4))
                for alg in self.keylist:
                    filepath = self.file_dir + item[alg] 
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
                if(self.file_name[self.file_path.index(item)][0]=='r'):
                    plt.xlim([0,2000])
                plt.legend(loc='lower right')
                fname = self.sat_file_dir + self.file_name[self.file_path.index(item)]
                print(fname)
                if save:
                    plt.savefig(fname)
                    print(">>> Figure saved at ", fname)
                    plt.close()
                    continue

                plt.show()
                    
            else:
                print("All dictionary values are empty.")



    def plot_curve(self, mode, save=False):
        for item in self.file_path:
            if self.keylist:
                print("Non-empty keys:", self.keylist)
                fig, ax = plt.subplots(figsize=(5,4))
                for alg in self.keylist:
                    filepath = self.file_dir + item[alg] 
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
                if(self.file_name[self.file_path.index(item)][0]=='r'):
                    plt.xlim([0,2000])
                fname = self.curve_file_dir + self.file_name[self.file_path.index(item)]
                print(fname)
                if save:
                    plt.savefig(fname)
                    print(">>> Figure saved at ", fname)
                    plt.close()
                    continue
                
                plt.show()
                    
            else:
                print("All dictionary values are empty.")


def readLog(file_path):
    '''
    read .Mat file and return data for analysis
    '''
    mat = loadmat(file_path)
    
    return mat["train_log"], mat["test_log"]


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

def get_non_empty_keys(dictionary):
    non_empty_keys = []
    for key, value in dictionary.items():
        if value != "":
            non_empty_keys.append(key)
    return non_empty_keys
    


app = App(file_path, file_name, file_dir='../data/')
app.plot_curve("Seg", save=True)
app.plot_reward("Seg", save=True)