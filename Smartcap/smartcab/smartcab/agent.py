import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import namedtuple, defaultdict
import numpy as np
import matplotlib.pyplot as plt

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.listOfActions = ["forward", "right", "left", None]
        self.state = []                 # State of the agent
        self.qTable = defaultdict(int)  # Q values
        self.alpha = 1.0                # Learning rate
        self.gamma = 0.0                # Gamma value
        self.temperature = 2.0         # Softmax temperature
        self.trial = 0                  # Trial number
        self.reward_trial = 0.0         # Cumulative reward on current trial
        self.rewards = []               # List of total reward per trial
        self.reach_dest = [0] * 100      # Number of times the agent reaches destination

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = []
        self.trial += 1
        self.alpha = 1.0 / float(self.trial)
        # Decrease temperature linearly, so that it reaches near 0 at the end of 100 trials
        self.temperature -= 0.0199
        self.alpha = 1.0/float(self.trial)

        # Store Cumulative reward on this trial and reset
        if self.trial > 1:
            self.rewards.append(self.reward_trial)
            self.reward_trial = 0.0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        coded_state = namedtuple("coded_state", ["light", "left", "oncoming", "next_waypoint"])
        s = coded_state(light=inputs['light'], left=inputs['left'], oncoming=inputs['oncoming'],
                        next_waypoint=self.next_waypoint)

        self.state = s
        q_values = []
        for a in self.listOfActions:
            if (self.state, a) not in self.qTable:
                self.qTable[(self.state, a)]  # = 10.0
            q_values.append(self.qTable[self.state, a])
        
        # TODO: Select action according to your policy
        # 1. Basic diving agent: random actions
        # action = random.choice(self.listOfActions)

        # 2. Naive agent
        # action = self.next_waypoint

        # 3. greedy policy following Q values
        # action = self.listOfActions[np.argmax(q_values)]

        # 4. Softmax selection
        dist = softmax(q_values, self.temperature)
        r = random.random()
        for idx, val in enumerate(self.listOfActions):
            if r < sum(dist[:idx+1]):
                action = val
                break

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Storing the rewards for analysis
        self.reward_trial += reward
        # If agent reaches destination
        if self.env.done:
            self.reach_dest[self.trial-1] = 1

        # TODO: Learn policy based on state, action, reward
        # Check new state
        inputs = self.env.sense(self)
        new_state = coded_state(light=inputs['light'], left=inputs['left'], oncoming=inputs['oncoming'],
                                next_waypoint=self.planner.next_waypoint())

        # Check best action from current state
        q_values = []
        for a in self.listOfActions:
            if (new_state, a) not in self.qTable:
                self.qTable[(new_state, a)]  # = 10.0
            q_values.append(self.qTable[new_state, a])

        # Update Q-value
        self.qTable[(self.state, action)] = (float(1-self.alpha)*self.qTable[(self.state, action)] +
                                             self.alpha*(reward + self.gamma*np.max(q_values)))

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

def softmax(w, t):
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.05, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    num_trials = 100
    sim.run(n_trials=num_trials)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    # Success rate in the last 20 trials
    print "Agent reaches destination {}% of the time during the last 20 trials".format(sum(a.reach_dest[80:])*100 / float(20.0))
    print "Agent reaches destination {}% of the time".format(sum(a.reach_dest)*100 / float(100.0))

    # plt.plot(a.rewards)
    # plt.ylabel('Reward')
    # plt.xlabel('Trial')
    # plt.show()

if __name__ == '__main__':
    run()
