import random
import itertools
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, **kwargs):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'black'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.Q = dict() #list of tuples (a, b) where is the states and b are the actions
        if kwargs:
            self.gamma = kwargs['gamma'] #discount factor 0.8                
            self.eps_0 = kwargs['eps_0']
        else:
            self.gamma = 0.7
            self.eps_0 = 0.8                        
        self.t = 1 #time initialization
        self.alpha = 1/self.t #learning rate
        self.action_idxs = zip(range(0,4), Environment.valid_actions)        
        self.trial = 0
        self.total_trials = 100    
        self.initialize_Q()   
        
             
        # TODO: Initialize any additional variables here

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.trial +=1 #updates every trial   
        self.calculateEpsilon() # eps decay after every trial
        self.numPenalties = 0
        # TODO: Prepare for a new trip; reset any variables here, if required
    
    #implementation of epsilon greedy algorithm 
    def calculateEpsilon(self):
        eps = self.eps_0 - self.trial*(self.eps_0)/self.total_trials
        return eps                                        
        
    #generates a state vector with random values        
    def randomInit(self):
        return [0.5*random.random() for i in xrange(4)]     
        
    #Q learning matrix randomly initialized
    def initialize_Q(self):        
#         'light', 'oncoming', 'left', 'next_waypoint'
        s = [['red', 'green'], ['left', 'right','forward', None],\
         ['left', 'right','forward', None], \
        ['right', 'left', 'forward'] ]
        listOfStates = list(itertools.product(*s))
        
        for state in listOfStates:
            self.Q[state] = self.randomInit()
                                              
    def update(self, t):
        self.t += 1 #updates at every step       
        self.alpha = 1./self.t
        
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator        
#        deadline = self.env.get_deadline(self)

        # TODO: Update state
        curr_state = self.env.sense(self)      
        curr_state['next_waypoint'] = self.planner.next_waypoint()

        #state <- ['light', 'oncoming', 'left', 'right', 'next_waypoint']        
        # TODO: Select action according to your policy
        #chooses a random action if random is lesser than currect epsilon
        #otherwise chooses a learned action from Q
        # Execute action and get reward        
                              
        q_values = self.Q[curr_state['light'], curr_state['oncoming'],
                   curr_state['left'],curr_state['next_waypoint']]

        if random.random() < self.calculateEpsilon():
            action = random.choice(Environment.valid_actions)
            action_idx = Environment.valid_actions.index(action)

        else:                                   
            action_idx = q_values.index(max(q_values)) #index of the action taken 
            #['light', 'oncoming', 'left', 'right', 'next_waypoint']
            action = Environment.valid_actions[action_idx]   
                                    
        reward = self.env.act(self, action)
                
    ################ Q-learning equation ############################
    #Q (state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]
    
    #Q(1, 5) = R(1, 5) + 0.8 * Max[Q(5, 1), Q(5, 4), Q(5, 5)] = 100 + 0.8 * 0 = 100
    #Q-learning equation V->(1-alpha)V + alpha(X)         
    #################################################################            
    # TODO: Learn policy based on state, action, reward 
    
        #calculates the future state in a separate variable
        future_state = self.env.sense(self)
        future_state['next_waypoint'] =  self.planner.next_waypoint()
        q_values_future = self.Q[future_state['light'], future_state['oncoming'],
                 future_state['left'], future_state['next_waypoint']]

        q_values[action_idx] = (1 - self.alpha) * (q_values[action_idx]) +\
            (self.alpha) * (reward + self.gamma*max(q_values_future))    
            
        self.Q[curr_state['light'], curr_state['oncoming'],
                curr_state['left'],curr_state['next_waypoint']] = q_values #updates Q-learning matrix                                
                        
                  
#        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    e = Environment()  #create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent, gamma = 0.2, eps_0 = 0.5)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0001, display = False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=a.total_trials)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()

#==============================================================================
# Debug Code
#    gammaSet = [0.2, 0.4, 0.6, 0.8]
#    epsSet = [0.3, 0.5, 0.7, 0.9]
#    results = list()
#    iterLen = 10
#    import timeit
#    start = timeit.timeit()
#    for gamma in gammaSet:
#        for eps_0 in epsSet:    
#            perf_tracker = list()
#            penalty_tracker = list()
#            for k in xrange(iterLen): # runs X times 100 trials and records 
#                e = Environment()  #create environment (also adds some dummy traffic)
#                a = e.create_agent(LearningAgent, gamma = gamma, eps_0 = eps_0)  # create agent
#                e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
#                # NOTE: You can set enforce_deadline=False while debugging to allow longer trials
#            
#                # Now simulate it
#                sim = Simulator(e, update_delay=0.0001, display = False)  # create simulator (uses pygame when display=True, if available)
#                # NOTE: To speed up simulation, reduce update_delay and/or set display=False
#            
#                sim.run(n_trials=a.total_trials)  # run for a specified number of trials
#                # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
#
#                #records the performance of the last 10 trials into perf_tracker                
#                perf_tracker.append(sim.perf)
#                
#                                            
#            print perf_tracker        
#            print penalty_tracker
#            print "total score: " + str(sum(perf_tracker)/iterLen)
#            
#            results.append((gamma, eps_0, sum(perf_tracker)/iterLen ))
#            
#    print results
#    end = timeit.timeit()
#    print "elapsed time: " + str(-start  + end)        
#==============================================================================


