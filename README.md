# RL Experiments 
# This repository contains a series of Reinforcement Learning experiments implemented in Python (Google Colab). Each experiment builds on core RL concepts, from setting up environments to applying advanced learning algorithms.

Experiment 1 — Simple Grid-World Environment
Objective: Build and interact with a simple grid-world environment using OpenAI Gymnasium's Taxi-v3.
What was done: The Taxi-v3 environment was loaded and explored — it has 500 possible states and 6 actions. A basic Q-learning loop was run for 2000 episodes using a learning rate (alpha) of 0.618. The agent's performance was tracked across episodes.
Conclusion: With a purely greedy (non-random) initial policy, the agent scored a total reward of -1280 in the first episode, showing it had no useful knowledge yet. After training for 2000 episodes, the agent consistently achieved positive rewards (ranging from 3 to 12 per episode), confirming it had successfully learned an optimal policy through Q-learning. The agent went from completely lost to reliably solving the task.

Experiment 2 — Upper Confidence Bound (UCB) Algorithm
Objective: Compare the UCB algorithm against random selection in a 10-armed bandit setting using an ad click dataset.
What was done: Random selection was first run over 10,000 rounds, selecting ads without any strategy. Then the UCB algorithm was implemented, which balances exploration and exploitation using a confidence bound that shrinks as each arm is pulled more.
Conclusion: Random selection achieved a total reward of 1258, while UCB achieved 2125 — nearly 69% higher. This clearly demonstrates that UCB's principled exploration strategy is far more efficient than random guessing. By maintaining an upper confidence estimate for each arm, UCB figures out which ad is the best much faster and exploits that knowledge effectively.

Experiment 3 — MDP and Policy Iteration
Objective: Model a 4x4 grid-world as a Markov Decision Process (MDP) and find the optimal policy using Policy Iteration.
What was done: A 4x4 grid was defined with two terminal states — (0,0) with reward 0 and (3,3) with reward 1. The policy iteration algorithm alternated between policy evaluation (computing state values under the current policy) and policy improvement (updating the policy to be greedy with respect to those values) until convergence.
Conclusion: Policy iteration successfully converged to an optimal policy. States closer to the goal (3,3) received higher value estimates (up to 1.0), and states farther away had lower values, correctly reflecting the discounted future reward. The resulting policy directed the agent to move right or down depending on its position, leading it straight to the goal. This confirms that policy iteration is a reliable method for solving finite MDPs when the environment model is known.

Experiment 4 — Dynamic Programming on FrozenLake
Objective: Apply dynamic programming (policy evaluation and policy improvement) to the FrozenLake-v1 environment from OpenAI Gym.
What was done: The FrozenLake environment was set up — a slippery grid world where the agent must reach the goal without falling into holes. Policy evaluation and policy improvement steps were applied using the environment's transition model to compute optimal state values and extract the best policy.
Conclusion: Dynamic programming effectively computed the value function for FrozenLake and derived a policy that navigates the agent toward the goal. Because the environment is stochastic (the agent doesn't always move in the intended direction), the values reflected the probabilistic nature of the transitions. This experiment reinforced that DP methods work well when the full model of the environment (transition probabilities and rewards) is available.

Experiment 5 — Monte Carlo Control and Temporal Difference Learning
Objective: Train a reinforcement learning agent in a 4x4 grid-world using three different algorithms — Monte Carlo Control, SARSA (on-policy TD), and Q-Learning (off-policy TD).
What was done: A custom GridWorld environment was built where the agent starts at (0,0) and must reach the goal at (3,3). All three algorithms were trained for 5000 episodes each using epsilon-greedy exploration. Q-values were compared across all three methods.
Conclusion: SARSA and Q-Learning both learned meaningful Q-values that clearly increased as states got closer to the goal. Q-Learning showed slightly more aggressive exploitation due to its off-policy nature (using max Q for updates), while SARSA's values were more conservative, reflecting the actual behavior policy. Monte Carlo showed zeros for many early states in the sampled output, which is expected since it only updates at the end of episodes and early states may be visited infrequently. All three algorithms are valid approaches, but Q-Learning tends to converge to the optimal policy faster in deterministic environments.

Experiment 6 — Exploration Strategies in Multi-Armed Bandit
Objective: Compare Epsilon-Greedy and UCB1 strategies on a 10-armed bandit over 1000 steps across 200 independent runs.
What was done: A MultiArmedBandit class was built with true reward values sampled from a normal distribution. An Agent class implemented both epsilon-greedy (with ε=0.1) and UCB1 (with c=2) strategies. Average reward per step was plotted over 1000 steps to compare convergence behavior.
Conclusion: UCB1 outperformed Epsilon-Greedy consistently across all phases. In the early phase, UCB1 found better arms faster. In the middle and final phases, UCB1 stabilized at a higher average reward because its confidence-based exploration naturally reduces exploration once it's certain about the best arm. Epsilon-Greedy, by contrast, keeps randomly exploring 10% of the time forever, creating a permanent performance floor. This confirms that UCB1's dynamic, uncertainty-driven exploration is more efficient than the static random exploration of Epsilon-Greedy, especially as the number of steps increases.
