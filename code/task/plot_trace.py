# %%
from core.utils import *
from core.world import *
import argparse

parser = argparse.ArgumentParser(description='Input tau/gridsize')
parser.add_argument('--dir', type=str, help='directory of the log file, e.g. ../data/reaching/', default='../data/reaching/')
parser.add_argument('--tau', type=int, help='length of the subtask, e.g. tau = 5', default=5)
parser.add_argument('--gridsize', type=int, help='size of the grid world', default=5)
parser.add_argument('--suffix', type=str, help='suffix of the log file', default='0827a')
parser.add_argument('--alg', type=str, help="form naive, naiveER, CFER, CFPER",default='naive')


args = parser.parse_args()
# read data

file_dir = args.dir
alg = args.alg
tau = args.tau 
grid_size = args.gridsize
suffix = args.suffix



file_name = f'{alg}_tau{tau}_grid{grid_size}_{suffix}.mat'
filepath = file_dir + file_name

test_log, train_log = readLog(filepath)



obstacle = np.array([[0,0.2,0.2,0.2],[0,0.4,0.4,0.2],[0.4,0,0.2,0.2],[0.6,0.4,0.4,0.2],[0.4,0.8,0.2,0.2]])
region = np.array([[0.6,0,0.4,0.4],[0,0.6,0.4,0.4],[0.6,0.6,0.4,0.4]])
if grid_size == 5:
    s0 = np.array([0.3, 0.1])
elif grid_size == 10:
    s0 = np.array([0.35,0.15])

world = World(grid_size=grid_size, 
              region=region,  
              prob_right=0.91, 
              beta=50,scale=[1,-0.01],obstacle=obstacle)




plotSampleTrace(world, test_log)

_ = input("Press [enter] to continue.")