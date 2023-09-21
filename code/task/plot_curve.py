# %%
from core.utils import *
import argparse

parser = argparse.ArgumentParser(description='Input tau/gridsize')
parser.add_argument('--dir', type=str, help='directory of the log file, e.g. ../data/reaching/', default='../data/reaching/')
parser.add_argument('--set', type=str, help='setting of the experiment, e.g. tau5_gird5', default='tau5_grid5')
parser.add_argument('--suffix', type=str, help='suffix of the log file', default='0828')


args = parser.parse_args()
# read data

file_dir = args.dir
setting = args.set
suffix = args.suffix


prefixes = ['naive', 'naiveER', 'CFER', 'CFPER']

test_logs = []
train_logs = []

for prefix in prefixes:
    # print(prefix)
    filepath = file_dir + prefix + '_' + setting + '_' + suffix + '.mat'
    test_log, train_log = readLog(filepath)
    # print(test_log)
    test_logs.append(test_log)
    train_logs.append(train_log)
    
colors = ["#C848b9","#F962A7","#FD836D","#FFBA69"]

# plot learning curve with plotly

fig, ax = plt.subplots(figsize=(5,5))
window_size = 50

for i in range(4):
    # print(file_paths[i])
    data = getLearningCurveData(test_logs[i])
    # smoothed_arr, lower_bounds, upper_bounds = smoothData(data, 100)
    smoothed_arr, lower_bounds, upper_bounds = curveData(data, window_size)
    # plot smoothed data
    x = np.array(range(len(smoothed_arr)))*window_size
    plt.plot(x, smoothed_arr, c=colors[i],linewidth=2, label=prefixes[i])
    
    plt.fill_between(x, lower_bounds, upper_bounds, facecolor=colors[i], alpha=0.2)
# plt.ylim([0,1])
plt.legend()
plt.show(block=False)

_ = input("Press [enter] to continue.")