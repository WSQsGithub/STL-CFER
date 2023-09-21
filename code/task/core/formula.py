from core.utils import *

class Formula: 
    def __init__(self, op_out, op_in, T, tau) -> None:
        '''
        initialize instance with number of subformulas, the outer temporal operator and the subformula operators and horizons
        '''
        self.n_flag = len(op_in) # number of flag variable needed
        self.op_out = op_out # the outer temporal operator:G/F
        self.op_in = op_in # the inner temporal opertator for each subtask: [G,F]
        self.flag = np.zeros(self.n_flag) # each flag variable
        self.T = T # time bound for outer temporal operator
        self.tau = tau # horizon for each subtask
        self.H = self.T + np.max(self.tau) # horizon of the overall formula
        
    def showFormula(self) -> None:
        self.formula = self.op_out + f'[0,{self.T})('
        for i in self.n_flag:
            self.formula += self.op_in[i] + f'[0,{self.tau[i]}) phi_{i} '
        self.formula += ')'
        print(self.formula)
        
        
    def updateFlag(self, sat: list, flag: int) -> int:
        '''
        sat: a list that contains the true value of each subformula's predicate
        e.g. task.updateFlag(task.sat4Pred(state))
        '''
        new_flag = np.zeros(self.n_flag)
        for i in range(0, self.n_flag):
            if self.op_in[i] == 'G':
                if sat[i] == 1:
                    new_flag[i] = min(flag[i]+1, self.tau[i])
                else:
                    new_flag[i] = 0
            elif self.op_in[i] == 'F':
                if sat[i] == 0:
                    new_flag[i] = max(flag[i]-1, 0)
                else:
                    new_flag[i] = self.tau[i]
        return new_flag
  
    def sat4Sub(self, flag) -> bool:
        '''
        return the satisfaction for each subformula as a list
        '''
        sat_sub = np.zeros(self.n_flag)
        for i in range(self.n_flag):
            if self.op_in[i] == 'G':
                sat_sub[i] = (flag[i]==self.tau[i])
            elif self.op_in[i] == 'F':
                sat_sub[i] = (flag[i]>0)
        return sat_sub
    
           
class Formula_task_FG(Formula):
    # phi = F[0,12](G[0,3] x in A)
    def __init__(self, op_out, op_in, T, tau, target):
        super().__init__(op_out, op_in, T, tau)
        self.A = np.array([target[0:2],target[0:2]+target[2:4]])
        
    def sat4Pred(self,state) -> list:
        '''
        define the satisfaction of each predicate in the subformulas
        return a list of n_flag variables
        '''
        sat  = np.zeros(self.n_flag)
        # x in A
        sat[0] = np.min(state>self.A[0,:]) and np.min(state<self.A[1,:])
        return sat
        
    def sat4All(self, flag) -> int:
        '''
        define the satisfaction for the whole inner formula
        '''
        return np.min(self.sat4Sub(flag))
    
    def viewFlag(self):
        print('flag = ', self.flag, 'sat4sub = ', self.sat4Sub(), 'sat4All = ', self.sat4All()) 
        
    def getRobustness(self, trace) -> float:
        # the overall formula is F[0,T)G[0,tau[0]) x in A
        # print(np.shape(trace))
        r_sub = np.zeros(self.T)
        for t in range(self.T):
            r_sub[t] = np.min(np.minimum(np.min(trace[t:t+self.tau[0], :] - self.A[0,:],1), np.min(self.A[1,:] - trace[t:t+self.tau[0], :],1)))
        return np.max(r_sub)
        
        
class Formula_task_FF(Formula):
    # phi = F[0,T)(F[0,tau[0]) x in A)
    def __init__(self, op_out, op_in, T, tau, target):
        super().__init__(op_out, op_in, T, tau)
        self.A = np.array([target[0:2],target[0:2]+target[2:4]])
        
    def sat4Pred(self, state) -> list:
        sat = np.zeros(self.n_flag)
        sat[0] = np.min(state>self.A[0,:]) and np.min(state<self.A[1,:])
        return sat
    
    def sat4All(self, flag) -> int:
        return np.min(self.sat4Sub(flag))
    
    def viewFlag(self):
        print('flag = ', self.flag, 'sat4sub = ', self.sat4Sub(), 'sat4All = ', self.sat4All()) 
        
    def getRobustness(self, trace) -> float:
        r_sub = np.zeros(self.T)
        for t in range(self.T):
            r_sub[t] = np.max(np.minimum(np.min(trace[t:t+self.tau[0], :] - self.A[0,:],1), np.min(self.A[1,:] - trace[t:t+self.tau[0], :],1)))
        return np.max(r_sub)

    
class Formula_task_GF(Formula):
    # phi = G[0,T)(F[0,tau[0]) x in A and F[0,tau[1]) x in B)
    def __init__(self, op_out, op_in, T, tau, target):
        super().__init__(op_out, op_in, T, tau)
        self.A = np.array([target[0,0:2],target[0,0:2]+target[0,2:4]])
        self.B = np.array([target[1,0:2],target[1,0:2]+target[1,2:4]])
        
    def sat4Pred(self,state) -> list:
        '''
        define the satisfaction of each predicate in the subformulas
        return a list of n_flag variables
        '''
        sat  = np.zeros(self.n_flag)
        
        # x in A
        sat[0] = np.min(state>self.A[0,:]) and np.min(state<self.A[1,:])
        # x in B
        sat[1] = np.min(state>self.B[0,:]) and np.min(state<self.B[1,:])
        return sat
        
    def sat4All(self, flag) -> int:
        '''
        define the satisfaction for the whole inner formula
        '''
        return np.min(self.sat4Sub(flag))
    
    def viewFlag(self):
        print('flag = ', self.flag, 'sat4sub = ', self.sat4Sub(), 'sat4All = ', self.sat4All()) 

    def getRobustness(self, trace) -> float:
        # phi = G[0,T)(F[0,tau[0]) x in A and F[0,tau[1]) x in B)
        r_sub_A = np.zeros(self.T)
        r_sub_B  =np.zeros(self.T)
        r_sub = np.zeros(self.T)
        for t in range(self.T):
            # might be something wrong with it
            r_sub_A[t] = np.max(np.minimum(np.min(trace[t:t+self.tau[0], :] -self.A[0,:],1), np.min(self.A[1,:] - trace[t:t+self.tau[0], :],1)))
            r_sub_B[t] = np.max(np.minimum(np.min(trace[t:t+self.tau[1], :] -self.B[0,:],1), np.min(self.B[1,:] - trace[t:t+self.tau[1], :],1)))
            r_sub[t] = min(r_sub_A[t],r_sub_B[t])

            # _ = input("Press [enter] to continue.")
        return np.min(r_sub)