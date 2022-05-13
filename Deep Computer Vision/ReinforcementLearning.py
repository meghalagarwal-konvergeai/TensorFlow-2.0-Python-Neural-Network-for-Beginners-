'''
---> Reinforcement Learning

The next and final topic in this course covers Reinforcement Learning.
This technique is different than many of the other machine learning techniques we have seen earlier and has many applications in training agents (an AI) to interact with enviornments like games.
Rather than feeding our machine learning model millions of examples we let our model come up with its own examples by exploring an enviornemt.
The concept is simple. Humans learn by exploring and learning from mistakes and past experiences so let's have our computer do the same.

---> Terminology

Before we dive into explaining reinforcement learning we need to define a few key peices of terminology.
--> Enviornemt
Enviornemt In reinforcement learning tasks we have a notion of the enviornment.
This is what our agent will explore.
An example of an enviornment in the case of training an AI to play say a game of mario would be the level we are training the agent on.
--> Agent
Agent an agent is an entity that is exploring the enviornment.
Our agent will interact and take different actions within the enviornment.
In our mario example the mario character within the game would be our agent.
--> State
State always our agent will be in what we call a state.
The state simply tells us about the status of the agent.
The most common example of a state is the location of the agent within the enviornment.
Moving locations would change the agents state.
--> Action
Action any interaction between the agent and enviornment would be considered an action.
For example, moving to the left or jumping would be an action.
An action may or may not change the current state of the agent.
In fact, the act of doing nothing is an action as well! The action of say not pressing a key if we are using our mario example.
--> Reward
Reward every action that our agent takes will result in a reward of some magnitude (positive or negative).
The goal of our agent will be to maximize its reward in an enviornment.
Sometimes the reward will be clear, for example if an agent performs an action which increases their score in the enviornment we could say they've recieved a positive reward.
If the agent were to perform an action which results in them losing score or possibly dying in the enviornment then they would recieve a negative reward.

The most important part of reinforcement learning is determing how to reward the agent.
After all, the goal of the agent is to maximize its rewards.
This means we should reward the agent appropiatly such that it reaches the desired goal.

---> Q-Learning

Now that we have a vague idea of how reinforcement learning works it's time to talk about a specific technique in reinforcement learning called Q-Learning.

Q-Learning is a simple yet quite powerful technique in machine learning that involves learning a matrix of action-reward values.
This matrix is often reffered to as a Q-Table or Q-Matrix.
The matrix is in shape (number of possible states, number of possible actions) where each value at matrix[n, m] represents the agents expected reward given they are in state n and take action m.
The Q-learning algorithm defines the way we update the values in the matrix and decide what action to take at each state.
The idea is that after a succesful training/learning of this Q-Table/matrix we can determine the action an agent should take in any state by looking at that states row in the matrix and taking the maximium value column as the action.
'''

import gym
import numpy as np
import time

env = gym.make('FrozenLake-v0')  # we are going to use the FrozenLake enviornment
print(env.observation_space.n)   # get number of states
env.reset()  # reset enviornment to default state
action = env.action_space.sample()  # get a random action 
new_state, reward, done, info = env.step(action)  # take action, notice it returns information about the action
env.render()   # render the GUI for the enviornment 


print(env.action_space.n)   # get number of actions

env = gym.make('FrozenLake-v0')
STATES = env.observation_space.n
ACTIONS = env.action_space.n

Q = np.zeros((STATES, ACTIONS))
EPISODES = 1500 # how many times to run the enviornment from the beginning
MAX_STEPS = 100  # max number of steps allowed for each run of enviornment
LEARNING_RATE = 0.81  # learning rate
GAMMA = 0.96
RENDER = False # if you want to see training set to true
epsilon = 0.9

# # code to pick action
# if np.random.uniform(0, 1) < epsilon:  # we will check if a randomly selected value is less than epsilon.
#     action = env.action_space.sample()  # take random action
# else:
#     action = np.argmax(Q[state, :])  # use Q table to pick best action based on current values

# Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[new_state, :]) - Q[state, action])

rewards = []
for episode in range(EPISODES):

  state = env.reset()
  for _ in range(MAX_STEPS):
    
    if RENDER:
      env.render()

    if np.random.uniform(0, 1) < epsilon:
      action = env.action_space.sample()  
    else:
      action = np.argmax(Q[state, :])

    next_state, reward, done, _ = env.step(action)

    Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action])

    state = next_state

    if done: 
      rewards.append(reward)
      epsilon -= 0.001
      break  # reached goal

print(Q)
print(f"Average reward: {sum(rewards)/len(rewards)}:")
# and now we can see our Q values!

# we can plot the training progress and see how the agent improved
import matplotlib.pyplot as plt

def get_average(values):
  return sum(values)/len(values)

avg_rewards = []
for i in range(0, len(rewards), 100):
  avg_rewards.append(get_average(rewards[i:i+100])) 

plt.plot(avg_rewards)
plt.ylabel('average reward')
plt.xlabel('episodes (100\'s)')
plt.show()