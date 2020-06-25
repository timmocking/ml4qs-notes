# Lecture 7: Reinforcement learning

Reinforcement is a means of influencing the user through interventions/actions and learning simultaneously.

There are two actors: the **user** and the **agent**.

The agent is providing support and is the entity that we are trying to create. The agent can observe the state of the user at time point *t* and derive an action to perform based on this (*A<sub>t</sub>*). Based on the user behaviour, we obtain a reward *R<sub>t+1</sub>*.

 In an iterative fashion, we observe what actions work best to yield "good" rewards. 

We do not strive for immediate rewards, but rewards that accumulate in the future, called the **value function**. 

A **policy** maps a state to an action (when to do what). 

We of course want to learn what types of behaviours work best, but we simultaneously want to get good rewards. Thus, we should balance **exploration** and **exploitation**.

Many algorithms for reinforcement learning require the **Markov property** to hold, meaning the probability of ending up on in one state based on the preceding history. 

## Markov Decision Process

If the Markov property holds, we can calculate the probability of going from one state to the next, as well as calculating the expected reward by modelling it as a **Markov Decision Process (MDP)**.

A policy selects a probability of an action in a state. 



For a given policy *π*, the expected value in a state *s* is given as follows using the **state-value function**

![](https://latex.codecogs.com/gif.latex?v_%7B%5Cpi%7D%28s%2C%20a%29%20%3D%20%5Cvarepsilon_%7B%5Cpi%7D%20%5Cleft%20%5B%20G_%7Bt%7D%20%7C%20S_%7Bt%7D%20%3D%20s%20%5Cright%20%5D)

Where *e* is our expectation given that we follow policy *π* of a value function *G<sub>t</sub>* given that we are in state *S*. 

This tells us how good is it to be in state *S* provided that later we follow a policy *π* (resulting in the result given by the value function).



The expected return of a policy for an action *a* in state *S* is called the **action-value function**. 

![](https://latex.codecogs.com/gif.latex?q_%7B%5Cpi%7D%28s%2C%20a%29%20%3D%20%5Cvarepsilon_%7B%5Cpi%7D%20%5Cleft%20%5B%20G_%7Bt%7D%20%7C%20S_%7Bt%7D%20%3D%20s_%7Bt%7DA_%7Bt%7D%20%3D%20a%20%5Cright%20%5D)



We want to find policies with the highest state-value function over all states. Similarly we want to find and optimal action-value function. 



## SARSA

SARSA is an algorithm for learning an MDP. 

The function *Q(S<sub>t</sub>A<sub>t</sub>)* defines the learned action value function for a given policy *π*.

For SARSA, we pick our actions in the same way at each step, which is called **on-policy**.



## Q-Learning

During Q-learning, we don't perform the next action before updating Q-values but assume that we select the highest value in the next state. This is called an **off-policy** approach. 

In Q-learning, an action is selected using a certain action selection approach. To determine the value of a selected action, the value of the resulting state is taken, plus the value of the action with the highest value
from that resulting state (while our action selection approach does not necessarily take that action with the highest reward). 

For SARSA, the same action selection selection mechanism is used, also
to compute the action to select in the next state. This is called on-policy.



## Eligibility tracing

Because we only looked one step ahead, we could not put credit to actions that contributed longer in the past. This is where **eligibility tracing** comes in.  This can be used to put credit on states and actions that have been seen more often in the past. The estimated values of state-actions pairs are updated more rigorously when these are seen more often in the past.

This can be used for both SARSA as well as Q-learning.



## U-tree Algorithm

The U-tree algorithm can be used to discretise continuous data spaces. It considers all observed values for the features included in the state and orders them per feature. It tries to look at all possible splits in this ordered list and checks whether the rewards accompanying the state before and after the split are significantly different using a Kolmogorov Smirnov test. The split which is most different (provided that the p-value is lower than 0.05) is taken and used to create a tree to discretize the state space. This process continues as more observations are performed.





