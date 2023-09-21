from core.utils import *
from core.world import *
from core.formula import *


class Q_Agent:
    def __init__(
        self,
        name: str,
        task: Formula,
        s0: float,
        Qtable_file: str = None,
    ) -> None:
        """initialize the agent"""
        self.name = name
        self.task = task
        self.actions = ['stay', 'N', 'S', 'W', 'E', 'NW', 'SW', 'NE', 'SE']
        self.s0 = s0
        self.state = self.s0
        self.f_s0 = task.flag
        self.f_state = self.f_s0
        self.string_state = packState(self.state, self.f_state)

        if Qtable_file is None:
            self.Qtable = self.build_q_table()
        else:  # build initial q-table from file
            self.read_q_table(Qtable_file)
            
    def learn(self, 
              world : World, 
              options : dict, 
              algorithm : str,
              ) -> dict:
        # common options
        
        self.MAX_EPISODE = options['MAX_EPISODE']
        self.CHECK_INTERVAL = options['CHECK_INTERVAL']


        # learning parameters
        [self.EPS_START, self.EPS_END, self.EPS_DECAY] = options['EPS']
        [self.LR_START, self.LR_END, self.LR_DECAY] = options['LR']
        self.gamma = options['GAMMA']

        print(f'>>> See Tensorboard at ../tensorboard/{self.name}')
        self.buffer_size = options['buffer_size'] # size of the experience pool
        self.buffer = self.build_buffer() # replay buffer
        self.replay_period = options['replay_period']
        self.batch_size = options['batch_size']


        # initialize learning
        self.n_updates_done = 0
        self.n_episodes_done = 0
        
        self.update_Meta(self.EPS_START, self.LR_START)

        # naive q-learning without experience replay
        if algorithm == 'naive':
            train_log = self.learn_naive(world)
            
        # q-learning with experience replay
        else:

            
            # 1. q-learning with naive experience replay
            if algorithm == 'naiveER':
                
                train_log = self.learn_with_naive_ER(world)
            
            # 2. q-learning with counterfactual experience replay
            elif algorithm == 'CFER':
                
                train_log = self.learn_with_CFER(world)
                
            # 3. q-learning with prioritized counterfactual experience replay
            elif algorithm == 'CFPER':
                
                train_log = self.learn_with_CFPER(world)
        
        return train_log
            
            
    def update_Meta(self,eps: float,lr:float) -> None:
        '''update hyperparameters'''
        self.eps = eps
        self.lr = lr
        
    def reset_State(self) -> None:
        self.state = copy.deepcopy(self.s0)
        self.f_state = copy.deepcopy(self.f_s0)

    def get_State(self) -> float:
        '''return the current state of the agent'''
        return [copy.deepcopy(self.state), copy.deepcopy(self.f_state)]
    
    def update_State(self, new_state, new_f_state) -> None:
        '''update the current state information based on the observation'''
        self.state = copy.deepcopy(new_state)
        self.f_state = copy.deepcopy(new_f_state)
        self.string_state = packState(self.state, self.f_state)
    
    def choose_action(self,state,f_state) -> int:
        '''choose action based on e-greedy method'''
        query = packState(state,f_state)
        row = self.query_Qtable(query)
        max_indices = np.where(row == np.max(row))[0] 
        action_id = random.choice(max_indices)
        if random.random()>self.eps: 
            return action_id
        else:
            return random.choice(range(len(self.actions)))
        
    def query_Qtable(self,query) -> float:
        '''find value of current state. 
            create new items if there exists none'''
            
        if query in self.Qtable.index:
            # Extract the row data as a NumPy array
            row = self.Qtable.loc[query].values
            #print('row_query=',row)
        else:
            new_row = pd.Series(data=np.zeros(9), index=self.Qtable.columns, name=query)
            self.Qtable = self.Qtable._append(new_row)
            row = np.zeros(len(self.Qtable.columns))
        return row
        
        
    def view_Qtable(self, all=False):
        #display(self.Qtable)
        print('#entries = ',self.Qtable.shape[0])
        if self.Qtable.shape[0] <= 5:
            print(tabulate(self.Qtable.round(4), headers = 'keys', tablefmt = 'pretty'))
        elif all == True:
            print(tabulate(self.Qtable.round(4), headers = 'keys', tablefmt = 'pretty'))
        else:
            random_rows = np.random.choice(self.Qtable.shape[0], size=5, replace=False) # randomly display 5 row
            print(tabulate(self.Qtable.round(4).iloc[random_rows], headers = 'keys', tablefmt = 'pretty'))
        
   
    def update_Qtable(self, 
                      string_state:str, 
                      next_string_state:str, action_id:int, 
                      reward:int,
                      gamma:float)-> None:
        # print(self.Qtable.shape[0],string_state)
        
        q_val = self.query_Qtable(string_state)[action_id]
        # print('-------------------------------update_Qtable--------------------------------')
        # print('now_state=', string_state ,'q_val=',q_val)
        
        # q_val = self.Qtable.loc[string_state, self.actions[action_id]]
        q_next = self.query_Qtable(next_string_state)
        max_q_next = np.max(q_next)
        
        # print('next_state=', next_string_state, 'max=',np.max(q_next))

        new_q = q_val + self.lr * (reward + gamma * max_q_next - q_val)
        
        self.Qtable.loc[string_state, self.actions[action_id]] = new_q
        self.n_updates_done += 1
        
        
    def build_q_table(self):
        state_str = packState(self.state, self.f_state)

        return pd.DataFrame(
            np.zeros((1, len(self.actions))),
            columns=self.actions,
            index = [state_str]
        )
        
    def read_q_table(self, file_path):
        self.Qtable = pd.read_csv(file_path,index_col=0)
        self.view_Qtable()
    
    def new_q_table(self):
        self.Qtable = self.build_q_table()
    
    def save_q_table(self, file_path):
        try:
            self.Qtable.to_csv(file_path, index=True)
            print(f"Q table successfully saved: {file_path}")
        except Exception as e:
            print(f"Q table saving error: {str(e)}")
            
    def build_buffer(self):
        
        cols = ['S', 'A', 'R', 'S_next', 'p']
        return pd.DataFrame(
            columns=cols
        )
        
    def load_buffer(self, experience: dict):
        '''add new experience to the buffer and forget old ones'''
        exp = pd.DataFrame.from_dict(experience, orient = 'index').T
        self.buffer = self.buffer._append(exp, ignore_index=True)
        
        if len(self.buffer)>self.buffer_size:
            self.buffer = self.buffer.drop(self.buffer.index[:-self.buffer_size]).reset_index(drop=True)
                    
    
    def sample_buffer(self, n_sample:int, m:str='uniform') -> object:
        if m == 'uniform':
            random_rows = self.buffer.sample(n=n_sample,replace=True)
            return random_rows
        elif m == 'prioritized':
            # re-weighted sampling based on similarity score
            self.buffer['weight'] = self.buffer['p'] / self.buffer['p'].sum()
            
            random_rows = self.buffer.sample(n=n_sample,weights='weight' ,replace=True)
            return random_rows


    def generate_cf_experience(self, state, f_state, action, new_state, world) -> list:
        
        
        experience = dict()
        experience['S'] = []
        experience['A'] = []
        experience['R'] = [] 
        experience['S_next'] = []
        experience['p'] = []
        
        cf_states = list(itertools.product(*[range(t) for t in self.task.tau]))

        for cf_state in cf_states:
            cf_state = np.array(cf_state)
            
            new_cf_state = self.task.updateFlag(self.task.sat4Pred(new_state),cf_state) 
            reward = world.getReward(self.task.sat4All(new_cf_state),self.task) 
            
            string_state = packState(state, cf_state)
            next_string_state = packState(new_state, new_cf_state)
            
            # calculate similarity
            similarity = np.linalg.norm(cf_state-f_state, ord=1) # maybe try using L2 norm?
            p = np.exp(-similarity)
            
            experience['S'].append(string_state)
            experience['A'].append(action)
            experience['R'].append(reward)
            experience['S_next'].append(next_string_state)
            experience['p'].append(p)
              
        return experience
    
    def decay_meta(self):
        self.eps = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1*self.n_episodes_done/self.EPS_DECAY)
        self.lr = self.LR_END + (self.LR_START - self.LR_END) * math.exp(-1*self.n_episodes_done/self.LR_DECAY)

    def learn_naive(
        self,
        world: World
    ):
        try:
            # initialize a tensorboard directory
            writer = SummaryWriter(f"../tensorboard/{self.name}")
            # initialize a data collector
            
            train_log = dict()
            
            episode_reward_log = np.zeros((1,self.MAX_EPISODE))
            reward_log = np.zeros((self.task.H ,self.MAX_EPISODE))
            state_log = np.zeros((self.task.H ,2,self.MAX_EPISODE))
            f_state_log = np.zeros((self.task.H , self.task.n_flag,self.MAX_EPISODE))
            rb_log = np.zeros((1,self.MAX_EPISODE))

            p_bar = tqdm(range(self.MAX_EPISODE), desc="# episode")
            for episode in p_bar:
                # parameter decay

                self.decay_meta()

                p_bar.set_postfix_str(f"lr = {self.lr}, eps = {self.eps}")

                self.reset_State()
    
                episode_reward = 0
                
                # one learning iteraction
                for step in range(self.task.H):
                    # choose action from current state
                    state, f_state = self.get_State()
                    action = self.choose_action(state, f_state)

                    # interact with world and get new_state
                    new_state, action, end = world.getObservation(state,action)
                    
                    # update task progress 
                    new_f_state = self.task.updateFlag(self.task.sat4Pred(new_state), f_state)
                        
                    # determine reward
                    reward = world.getReward(self.task.sat4All(new_f_state),self.task)
                    
                    # print('states =', state, f_state, ', action=', self.actions[action], 'reward=', reward, ',new_state=', new_state, new_f_state)
                    
                    episode_reward += reward
                    
                    # update Q 
                    self.update_Qtable(packState(state,f_state), packState(new_state,new_f_state), action, reward, self.gamma)
                    # self.view_Qtable(all=True)
                    
                    reward_log[step,episode] = reward
                    state_log[step,:,episode] = state
                    f_state_log[step,:,episode] = f_state                    
                    # update self state
                    self.update_State(new_state,new_f_state)
                    
                    
                    

                
                episode_reward_log[0,episode] = episode_reward
                rb_log[0,episode] = self.task.getRobustness(state_log[:, :,episode])
                self.n_episodes_done += 1
                # show curve every check interval
                if ~episode%self.CHECK_INTERVAL and episode:
                    writer.add_scalar('naive', np.mean(episode_reward_log[0][episode-self.CHECK_INTERVAL: episode]))

        except KeyboardInterrupt:
            print('>>> Interupted at episode = ', episode)
            
        train_log['episode'] = episode+1
        train_log['episode_reward'] = episode_reward_log
        train_log['state'] = state_log
        train_log['f_state'] = f_state_log
        train_log['reward'] = reward_log
        train_log['rb'] = rb_log
        train_log['sat'] = (rb_log>0)
        
        return train_log
    
    # this comparison should not exist
    def learn_with_naive_ER(self, world)-> int:
        '''Learning loop with naive experience replay'''
        # print(batch_size)
        
        try: 
            # initialize a tensorboard directory
            writer = SummaryWriter(f"../tensorboard/{self.name}")
            # initialize a data collector
            
            train_log = dict()
            
            episode_reward_log = np.zeros((1,self.MAX_EPISODE))
            reward_log = np.zeros((self.task.H ,self.MAX_EPISODE))
            state_log = np.zeros((self.task.H ,2,self.MAX_EPISODE))
            f_state_log = np.zeros((self.task.H , self.task.n_flag,self.MAX_EPISODE))
            rb_log = np.zeros((1,self.MAX_EPISODE))
            
            p_bar = tqdm(range(self.MAX_EPISODE), desc="# episode")
            for episode in p_bar:
                
                # decaying learning parameters
                self.decay_meta()
                p_bar.set_postfix_str(f'lr = {self.lr}, eps = {self.eps}')
                
                self.reset_State()
                
                episode_reward = 0
                
                for step in range(self.task.H):
                    # choose action from current state
                    state, f_state = self.get_State()
                    action = self.choose_action(state, f_state)

                    # interact with world and get new_state    
                    new_state, action, end = world.getObservation(state,action)
                    
                    # update task progress 
                    new_f_state = self.task.updateFlag(self.task.sat4Pred(new_state), f_state)
                        
                    # determine reward
                    reward = world.getReward(self.task.sat4All(new_f_state),self.task)
                    
                    # print('states =', state, f_state, ', action=', self.actions[action], 'reward=', reward, ',new_state=', new_state, new_f_state)

                    episode_reward += reward
                    
                    # store transition in the experience pool
                    experience = dict()
                    experience['S'] = packState(state, f_state)
                    experience['A'] = action
                    experience['R'] = reward
                    experience['S_next'] = packState(new_state, new_f_state)
                    
                    self.load_buffer(experience)
                    
                    reward_log[step,episode] = reward
                    state_log[step,:,episode] = state
                    f_state_log[step,:,episode] = f_state                    
                    # update self state
                    self.update_State(new_state,new_f_state)
                    
                    
                    # rechieve experience from the pool and update q-table every K step
                    if step%self.replay_period == 0:
                        
                        # process the minibatch
                        
                        samples = self.sample_buffer(self.batch_size) # sample from the pool
                        for index, row in samples.iterrows():
                            string_state = row['S']
                            next_string_state = row['S_next']
                            r = row['R']
                            a = row['A']
                            # update Q
                            self.update_Qtable(string_state, next_string_state, a, r, self.gamma)    
                episode_reward_log[0,episode] = episode_reward
                rb_log[0,episode] = self.task.getRobustness(state_log[:, :,episode])
                self.n_episodes_done += 1


                # show curve every check interval
                if ~episode%self.CHECK_INTERVAL and episode:
                    writer.add_scalar('naive', np.mean(episode_reward_log[0][episode-self.CHECK_INTERVAL: episode]))
            
        except KeyboardInterrupt:
            print('>>> Interupted at episode = ', episode)
            
        train_log['episode'] = episode+1
        train_log['episode_reward'] = episode_reward_log
        train_log['state'] = state_log
        train_log['f_state'] = f_state_log
        train_log['reward'] = reward_log
        train_log['rb'] = rb_log
        train_log['sat'] = (rb_log>0)
        
        return train_log
    
    def learn_with_CFER(self, world)-> int:
        '''Learning loop with counterfactual experience replay'''
        # print(batch_size)
        
        try:
           # initialize a tensorboard directory
            writer = SummaryWriter(f"../tensorboard/{self.name}")
            
            # initialize a data collector
            
            train_log = dict()
            
            episode_reward_log = np.zeros((1,self.MAX_EPISODE))
            reward_log = np.zeros((self.task.H ,self.MAX_EPISODE))
            state_log = np.zeros((self.task.H ,2,self.MAX_EPISODE))
            f_state_log = np.zeros((self.task.H , self.task.n_flag,self.MAX_EPISODE))
            rb_log = np.zeros((1,self.MAX_EPISODE))

            p_bar = tqdm(range(self.MAX_EPISODE), desc="# episode")
            for episode in p_bar:
                
                # parameter decay
                self.decay_meta()
                p_bar.set_postfix_str(f"lr = {self.lr}, eps = {self.eps}")

                self.reset_State()
                
                episode_reward = 0
                
                for step in range(self.task.H):
                    # choose action from current state
                    state, f_state = self.get_State()
                    action = self.choose_action(state, f_state)

                    # interact with world and get new_state    
                    new_state, action, end = world.getObservation(state,action)
                    
                    # update task progress 
                    new_f_state = self.task.updateFlag(self.task.sat4Pred(new_state), f_state)
                        
                    # determine reward
                    reward = world.getReward(self.task.sat4All(new_f_state),self.task)
                    
                    # print('states =', state, f_state, ', action=', self.actions[action], 'reward=', reward, ',new_state=', new_state, new_f_state)

                    episode_reward += reward
                    
                    # update using real experience
                    self.update_Qtable(packState(state,f_state), packState(new_state,new_f_state), action, reward, self.gamma)
                    
                    # generate CF experience and store transition in the experience pool
                    self.load_buffer(self.generate_cf_experience(state, f_state, action, new_state, world))
                    
                    
                    reward_log[step,episode] = reward
                    state_log[step,:,episode] = state
                    f_state_log[step,:,episode] = f_state                    
                    # update self state
                    self.update_State(new_state,new_f_state)
                    
                    
                    
                    # rechieve experience from the pool and update q-table every K step
                    if step%self.replay_period == 0:
                        
                        # process the minibatch
                        
                        samples = self.sample_buffer(self.batch_size) # sample from the pool
                        for index, row in samples.iterrows():
                            string_state = row['S']
                            next_string_state = row['S_next']
                            r = row['R']
                            a = row['A']
                            # update Q
                            self.update_Qtable(string_state, next_string_state, a, r,self.gamma)    
                
                episode_reward_log[0,episode] = episode_reward
                rb_log[0,episode] = self.task.getRobustness(state_log[:, :,episode])
                self.n_episodes_done += 1

                # show curve every check interval
                if ~episode%self.CHECK_INTERVAL and episode:
                    writer.add_scalar('naive', np.mean(episode_reward_log[0][episode-self.CHECK_INTERVAL: episode]))
            
        except KeyboardInterrupt:
            print('>>> Interupted at episode = ', episode)
            
        train_log['episode'] = episode+1
        train_log['episode_reward'] = episode_reward_log
        train_log['state'] = state_log
        train_log['f_state'] = f_state_log
        train_log['reward'] = reward_log
        train_log['rb'] = rb_log
        train_log['sat'] = (rb_log>0)
        
        return train_log
    
    def learn_with_CFPER(self, world)-> int:
        '''Learning loop with counterfactual prioritized experience replay'''
        try:
            # initialize a tensorboard directory
            writer = SummaryWriter(f"../tensorboard/{self.name}")
            
            # initialize a data collector
            
            train_log = dict()
            
            episode_reward_log = np.zeros((1,self.MAX_EPISODE))
            reward_log = np.zeros((self.task.H ,self.MAX_EPISODE))
            state_log = np.zeros((self.task.H ,2,self.MAX_EPISODE))
            f_state_log = np.zeros((self.task.H , self.task.n_flag,self.MAX_EPISODE))
            rb_log = np.zeros((1,self.MAX_EPISODE))

            
            p_bar = tqdm(range(self.MAX_EPISODE), desc="# episode")
            for episode in p_bar:
                
                # parameter decay
                self.decay_meta()

                p_bar.set_postfix_str(f"lr = {self.lr}, eps = {self.eps}")

                self.reset_State()
                
                # get current state
                state, f_state = self.get_State() 
                
                
                episode_reward = 0
                
                for step in range(self.task.H):
                   # choose action from current state
                    state, f_state = self.get_State()
                    action = self.choose_action(state, f_state)

                    # interact with world and get new_state    
                    new_state, action, end = world.getObservation(state,action)
                    
                    # update task progress 
                    new_f_state = self.task.updateFlag(self.task.sat4Pred(new_state), f_state)
                        
                    # determine reward
                    reward = world.getReward(self.task.sat4All(new_f_state),self.task)
                    
                    # print('states =', state, f_state, ', action=', self.actions[action], 'reward=', reward, ',new_state=', new_state, new_f_state)

                    episode_reward += reward
                    
                    # update using real experience
                    self.update_Qtable(packState(state,f_state), packState(new_state,new_f_state), action, reward, self.gamma)
                    
                    # generate CF experience and store transition in the experience pool
                    
                    self.load_buffer(self.generate_cf_experience(state, f_state, action, new_state, world))
                    
                    reward_log[step,episode] = reward
                    state_log[step,:,episode] = state
                    f_state_log[step,:,episode] = f_state                    
                    # update self state
                    self.update_State(new_state,new_f_state)
                    
                    
                    # rechieve experience from the pool and update q-table every K step
                    if step%self.replay_period == 0:
                        
                        # process the minibatch
                        
                        samples = self.sample_buffer(self.batch_size, 'prioritized') # sample from the pool
                        for index, row in samples.iterrows():
                            string_state = row['S']
                            next_string_state = row['S_next']
                            r = row['R']
                            a = row['A']
                            # update Q
                            self.update_Qtable(string_state, next_string_state, a, r, self.gamma)    
                
                episode_reward_log[0,episode] = episode_reward
                rb_log[0,episode] = self.task.getRobustness(state_log[:, :,episode])
                self.n_episodes_done += 1

                # show curve every check interval
                if ~episode%self.CHECK_INTERVAL and episode:
                    writer.add_scalar('naive', np.mean(episode_reward_log[0][episode-self.CHECK_INTERVAL: episode]))
            
            
        except KeyboardInterrupt:
            print('>>> Interupted at episode = ', episode)
            
        train_log['episode'] = episode+1
        train_log['episode_reward'] = episode_reward_log
        train_log['state'] = state_log
        train_log['f_state'] = f_state_log
        train_log['reward'] = reward_log
        train_log['rb'] = rb_log
        train_log['sat'] = (rb_log>0)
        
        return train_log
    
    def evaluate(self, world, n_trials)->None:
        
        episode_reward_log = np.zeros((1, n_trials))
        reward_log = np.zeros((self.task.H, n_trials ))
        state_log = np.zeros((self.task.H,2, n_trials))
        f_state_log = np.zeros((self.task.H , self.task.n_flag, n_trials,))
        rb_log = np.zeros((1,n_trials))
        
        # simulation for testing
        for episode in tqdm(range(n_trials), desc='# episode'):

            # start from initial state
            self.reset_State()   
            
            # use deterministic policy, no update on q-table
            self.update_Meta(0,0)
            
            # get current state
            episode_reward = 0
            
            for step in range(self.task.H):
                # choose action from current state
                state, f_state = self.get_State()
                action = self.choose_action(state, f_state)
                
                # interact with the world to reach new state
                new_state, action,end = world.getObservation(state,action)
                new_f_state = self.task.updateFlag(self.task.sat4Pred(new_state), f_state)
                
                # get reward
                reward = world.getReward(self.task.sat4All(new_f_state), self.task)
                
                episode_reward += reward
                
                reward_log[step,episode] = reward
                state_log[step,:,episode,] = state
                f_state_log[step,:,episode,] = f_state
                
                # update state
                self.update_State(new_state,new_f_state)
                
                
            rb_log[0,episode] = self.task.getRobustness(state_log[:, :, episode])
            episode_reward_log[0,episode] = episode_reward
                
        log = dict()
        log['episode_reward'] = episode_reward_log
        log['reward'] = reward_log
        log['state']  =state_log
        log['f_state'] = f_state_log
        log['rb'] = rb_log
        log['p_sat'] = sum(rb_log[0]>0)/n_trials
        
        print(f"Satisfaction rate = {log['p_sat']}")
        
        return log    
    