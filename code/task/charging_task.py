# Task FF: Phi = F_[0,T)F_[0,tau) s in A
# initialization
import sys
import os
from core.agent import *
from core.world import *
from core.formula import *
import warnings
warnings.filterwarnings("ignore")

import argparse
parser = argparse.ArgumentParser(description='Input tau/gridsize')
parser.add_argument('--T', type=int, help='length of the overall task', default=10)
parser.add_argument('--tau', type=int, help='length of the sub-task', default=5)
parser.add_argument('--gridsize', type=int, help='size of the grid world', default=5)
parser.add_argument('--alg', type=str, help='naive/ER/CFER/CFPER', default='naive')
parser.add_argument('--suffix', type=str, help='a custom suffix to name your log data', default='')
parser.add_argument('--save', type=int, help='whether to save data', default=1)

args = parser.parse_args()

#获得传入的参数
# task parameter
tau = args.tau
grid_size = args.gridsize
alg = args.alg
T = args.T

print(f'-----------------------T={T}, tau={tau}, gridsize={grid_size}, algorithm={alg}--------------------------------')
folder_path = '../data/charging/'
file_name = f'{alg}_tau{tau}_grid{grid_size}_{args.suffix}'
savedir = os.path.join(folder_path, file_name)
print('>>> Data will be saved at', savedir)



obstacle = np.array([[0,0.2,0.2,0.2],[0,0.4,0.4,0.2],[0.4,0,0.2,0.2],[0.6,0.4,0.4,0.2],[0.4,0.8,0.2,0.2]])
region = np.array([[0.6,0,0.4,0.4],[0,0.6,0.4,0.4],[0.6,0.6,0.4,0.4]])
if grid_size == 5:
    s0 = np.array([0.3, 0.1])
elif grid_size == 10:
    s0 = np.array([0.35,0.15])

world = World(grid_size=grid_size, 
              region=region,  
              prob_right=0.93, 
              beta=50,scale=[2,-1],obstacle=obstacle)

# world.showWorld()

task = Formula_task_FG(op_out='F', 
                     op_in=['G'], 
                     T=T, tau=[tau], target=region[2,:])

agent = Q_Agent(name=f"charging_{args.suffix}",
              task=task,
              s0=s0)
agent.view_Qtable()

# common options
options = dict()
# options['n_episode'] = 1000
# options['check_interval'] = 50
# options['eps_decay'] = 0.995
# options['alpha_decay'] = 0.999
# options['min_eps'] = 1e-6
# options['min_alpha'] = 1e-6

# # 0903
# options['n_episode'] = 1000
# options['check_interval'] = 50
# options['eps_decay'] = 0.99
# options['alpha_decay'] = 0.99
# options['min_eps'] = 1e-6
# options['min_alpha'] = 1e-2

# options['buffer_size'] = 1000
# options['batch_size'] = 100
# options['replay_period'] = 4


# 0905

options['MAX_EPISODE'] = 4000
options['CHECK_INTERVAL'] = 50
options['EPS'] = [0.99, 0.01, 400]
options['LR'] = [0.01, 0.01, 300]
options['buffer_size'] = 1000 # size of the experience pool
options['replay_period'] = 4
options['batch_size'] = 64
options['n_test'] = 500
options['GAMMA'] = 0.999


print('>>> Started learning at ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

train_log = agent.learn(world, options, alg)

print('>>> Stopped at ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

print('>>>')

agent.view_Qtable()

print('>>> Started test at ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

test_log = agent.evaluate(world=world, n_trials=options['n_test'])

print('>>> Stopped test at ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

print('>>>')


if args.save == 1:
    savemat(savedir +'.mat',{'train_log': train_log, 'test_log' : test_log})
    agent.save_q_table(savedir+'.csv')
    print('>>> Log saved at ', savedir)

print('>>>')


