from core.utils import *


class World():
    def __init__(self, grid_size, region, prob_right,beta, scale=[1,0], obstacle = None):
        '''
        size of the world is 1 x 1, devided into n x n grids
        e.g. region = np.array([[.5,.5,.2,.2],[.1,.3,.1,.1]]) where [.5,.5] is the left-bottom coordinate for the first region and [.2,.2] is its width and height
        the same with the obstacle
        prob_right is the motion certainty
        beta is the LSE factor
        scale is a 1*2 list for rescale the reward, default is [1,0]
        motion uncertainty is consist of a list of action id:
        [ 0    1     2    3     4    5    6    7     8]
        [stay  N     S    W     E    NW   NE   SW    SE]
        which correspond to the move dir list
        '''
        self.grid_size = grid_size
        self.region = region
        self.prob_right = prob_right
        self.beta = beta
        self.scale = scale
        self.action = ['stay', 'N', 'S', 'W', 'E', 'NW', 'SW', 'NE', 'SE']
        self.motion_uncertainty = np.array([[0,0,0],
                                            [0,5,7],
                                            [0,6,8],
                                            [0,5,6],
                                            [0,7,8],
                                            [0,1,3],
                                            [0,2,3],
                                            [0,1,4],
                                            [0,2,4]])
        self.obstacle = obstacle 

        self.move_dir = np.array([[0,0],
                                  [0,1],
                                  [0,-1],
                                  [-1,0],
                                  [1,0],
                                  [-1,1],
                                  [-1,-1],
                                  [1,1],
                                  [1,-1]])
        
        self.dgrid = 1/self.grid_size
        self.color = ['#A8BCDA', '#9F5751', '#7FA362']
        
    def updateSize(self, grid_size):
        self.grid_size = grid_size
        self.dgrid = 1/self.grid_size

    def update_Meta(self,prob_right):
        self.prob_right = prob_right
        
    def getObservation(self, cur_state, action):
        '''return the next physical state'''
        if random.random()>self.prob_right: # action uncertainty
            uncertainty = random.choice(self.motion_uncertainty[action,:])
        else: 
            uncertainty = action

        # print('next_states = ',next_states) # ok
        next_state = cur_state + self.dgrid*self.move_dir[uncertainty,:]
        
        
        # stay before collision
        if self.checkCollision(next_state):
            res = copy.deepcopy(cur_state)
            # a = 0
            a = copy.deepcopy(action)
            end = 1
            # print('collide!', cur_state, self.action[action], next_state)
        else:
            res = copy.deepcopy(next_state)
            a = copy.deepcopy(action)
            end = 0
        return res, a, end
        
    def checkCollision(self,state)-> int:
        # print(state,state - self.obstacle[:,0:1])
        
        if (np.min(state)<=0 or np.max(state)>=1):
            return 1
        if (self.obstacle is not None):
            for obs in self.obstacle:
                obs_x, obs_y, obs_width, obs_height = obs
                obs_top = obs_y + obs_height
                obs_right = obs_x + obs_width
                x, y = state
                if obs_x <= x <= obs_right and obs_y <= y <= obs_top:
                    return 1
        return 0
        
    def getReward(self,sat,task) -> int:
        '''
        if scale = [a,b], reward = a*sat+b
        '''
        # for the case of reachability
        if task.op_out == 'F':
            # reward = np.exp(self.beta*sat)*self.scale[0]+self.scale[1] # but since it's binary we wrote it as
            reward = sat*self.scale[0]+self.scale[1]
        elif task.op_out == 'G':
            # reward = sat*self.scale[0]+self.scale[1]
            reward = -np.exp(-self.beta*sat)*self.scale[0]+self.scale[1]
        return reward
    
    def showWorld(self, trajectory=None)-> None:

        # create a figure
        fig, ax = plt.subplots(figsize=(5,5))
        
        
        # plot grid
        for i in range(self.grid_size+1):
            ax.axhline(y=i*self.dgrid, color='black', linewidth=0.5)
            ax.axvline(x=i*self.dgrid, color='black', linewidth=0.5)
            
        # plot region
        for i in range(len(self.region)):
            left = self.region[i,0]
            bottom = self.region[i,1]
            width = self.region[i,2]
            height = self.region[i,3]
            rect = patches.Rectangle((left, bottom), width, height, facecolor=self.color[i])
            ax.add_patch(rect)
            
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        
        # plot obstacles
        if self.obstacle is not None:
            for i in range(len(self.obstacle)):
                left = self.obstacle[i,0]
                bottom = self.obstacle[i,1]
                width = self.obstacle[i,2]
                height = self.obstacle[i,3]
                rect = patches.Rectangle((left, bottom), width, height, facecolor='gray')
                ax.add_patch(rect)
            
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
    
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        if trajectory is None:
            plt.show(block=False)
            return
        else:
            start_point = trajectory[0]
            plt.plot(trajectory[:, 0], trajectory[:, 1], linewidth=4)
            plt.scatter(trajectory[:, 0], trajectory[:,1], marker='.', s=200, label='Start')
            plt.scatter(start_point[0], start_point[1], marker='*', color='red', s=200, label='Start')
            plt.show(block=False)
    
    