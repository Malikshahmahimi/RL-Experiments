# 🤖 RL Experiments — BE AIDS Sem 8

Hey! This repo contains my Reinforcement Learning experiments from college — covering everything from grid-worlds and bandit algorithms to MDP, policy iteration, Monte Carlo, SARSA, Q-Learning, function approximation, and deep Q-networks. All coded up in Python using NumPy, OpenAI Gym, PyTorch, and Matplotlib. Feel free to explore!

---

## 📋 Table of Contents
- [Exp 1 — Simple Grid-World Environment](#exp-1--simple-grid-world-environment)
- [Exp 2 — Upper Confidence Bound Algorithm](#exp-2--upper-confidence-bound-algorithm)
- [Exp 3 — MDP and Policy Iteration](#exp-3--mdp-and-policy-iteration)
- [Exp 4 — Dynamic Programming on FrozenLake](#exp-4--dynamic-programming-on-frozenlake)
- [Exp 5 — Monte Carlo Control and TD Learning](#exp-5--monte-carlo-control-and-td-learning)
- [Exp 6 — Exploration Strategies in Multi-Armed Bandit](#exp-6--exploration-strategies-in-multi-armed-bandit)
- [Exp 7 — Function Approximation in RL](#exp-7--function-approximation-in-rl)
- [Exp 8 — Deep Q-Network (DQN) on Atari Pong](#exp-8--deep-q-network-dqn-on-atari-pong)

---

## Exp 1 — Simple Grid-World Environment

**🎯 Objective:** Build and interact with a simple grid-world environment using OpenAI Gymnasium's Taxi-v3.

**🔧 What was done:**
The Taxi-v3 environment was loaded and explored — it has 500 possible states and 6 actions. A basic Q-learning loop was run for 2000 episodes using a learning rate (alpha) of 0.618. The agent's performance was tracked across episodes.

**✅ Conclusion:**
With a purely greedy initial policy, the agent scored a total reward of **-1280** in the first episode, showing it had no useful knowledge yet. After training for 2000 episodes, the agent consistently achieved positive rewards (ranging from **3 to 12** per episode), confirming it had successfully learned an optimal policy through Q-learning. The agent went from completely lost to reliably solving the task.

---

## Exp 2 — Upper Confidence Bound Algorithm

**🎯 Objective:** Compare the UCB algorithm against random selection in a 10-armed bandit setting using an ad click dataset.

**🔧 What was done:**
Random selection was first run over 10,000 rounds, selecting ads without any strategy. Then the UCB algorithm was implemented, which balances exploration and exploitation using a confidence bound that shrinks as each arm is pulled more.

**✅ Conclusion:**
Random selection achieved a total reward of **1258**, while UCB achieved **2125** — nearly **69% higher**. This clearly demonstrates that UCB's principled exploration strategy is far more efficient than random guessing. By maintaining an upper confidence estimate for each arm, UCB figures out which ad is best much faster and exploits that knowledge effectively.

---

## Exp 3 — MDP and Policy Iteration

**🎯 Objective:** Model a 4x4 grid-world as a Markov Decision Process (MDP) and find the optimal policy using Policy Iteration.

**🔧 What was done:**
A 4x4 grid was defined with two terminal states — (0,0) with reward 0 and (3,3) with reward 1. The policy iteration algorithm alternated between policy evaluation and policy improvement until convergence.

**✅ Conclusion:**
Policy iteration successfully converged to an optimal policy. States closer to the goal **(3,3)** received higher value estimates (up to **1.0**), and states farther away had lower values, correctly reflecting the discounted future reward. This confirms that policy iteration is a reliable method for solving finite MDPs when the environment model is known.

---

## Exp 4 — Dynamic Programming on FrozenLake

**🎯 Objective:** Apply dynamic programming (policy evaluation and policy improvement) to the FrozenLake-v1 environment from OpenAI Gym.

**🔧 What was done:**
The FrozenLake environment was set up — a slippery grid world where the agent must reach the goal without falling into holes. Policy evaluation and improvement steps were applied using the environment's transition model.

**✅ Conclusion:**
Dynamic programming effectively computed the value function for FrozenLake and derived a policy that navigates the agent toward the goal. Because the environment is stochastic, the values reflected the probabilistic nature of the transitions. This experiment reinforced that DP methods work well when the full environment model is available.

---

## Exp 5 — Monte Carlo Control and TD Learning

**🎯 Objective:** Train an RL agent in a 4x4 grid-world using three algorithms — Monte Carlo Control, SARSA (on-policy TD), and Q-Learning (off-policy TD).

**🔧 What was done:**
A custom GridWorld environment was built where the agent starts at (0,0) and must reach the goal at (3,3). All three algorithms were trained for 5000 episodes each using epsilon-greedy exploration. Q-values were compared across all three methods.

**✅ Conclusion:**
SARSA and Q-Learning both learned meaningful Q-values that increased as states got closer to the goal. Q-Learning showed slightly more aggressive exploitation due to its off-policy nature, while SARSA's values were more conservative. Monte Carlo showed zeros for many early states, expected since it only updates at episode end. **Q-Learning tends to converge to the optimal policy fastest** in deterministic environments.

---

## Exp 6 — Exploration Strategies in Multi-Armed Bandit

**🎯 Objective:** Compare Epsilon-Greedy and UCB1 strategies on a 10-armed bandit over 1000 steps across 200 independent runs.

**🔧 What was done:**
A `MultiArmedBandit` class was built with true reward values sampled from a normal distribution. An `Agent` class implemented both epsilon-greedy (ε=0.1) and UCB1 (c=2) strategies. Average reward per step was plotted over 1000 steps to compare convergence behavior.

**✅ Conclusion:**
UCB1 outperformed Epsilon-Greedy consistently across all phases. UCB1 stabilized at a higher average reward because its confidence-based exploration naturally reduces once it's certain about the best arm. Epsilon-Greedy keeps randomly exploring **10% of the time forever**, creating a permanent performance floor. **UCB1's dynamic exploration is simply smarter.**

---

## Exp 7 — Function Approximation in RL

**🎯 Objective:** Use function approximation techniques (linear regression) to approximate value functions in a reinforcement learning problem, instead of maintaining a lookup table.

**🔧 What was done:**
A 5-state Random Walk environment (states A–E with terminal states at both ends) was set up. States were encoded using one-hot feature vectors, and a linear value function `V(s) = w · x(s)` was trained using **TD(0) with function approximation** over 100 episodes. The learned weights were used to estimate state values, which were then compared against the analytically known true values.

**✅ Conclusion:**
The linear function approximator successfully converged toward the true value function after training. The estimated values closely tracked the true values (ranging from 1/6 to 5/6), demonstrating that even a simple linear model with TD learning can generalize across states. This experiment highlights how **function approximation scales RL to problems where tabular methods are impractical**.

---

## Exp 8 — Deep Q-Network (DQN) on Atari Pong

**🎯 Objective:** Implement a Deep Q-Network (DQN) to train an agent to play Atari Pong using raw pixel inputs.

**🔧 What was done:**
The full DQN pipeline was built using **PyTorch** and **OpenAI Gymnasium's ALE/Pong-v5** environment. This included:
- A **CNN-based DQN** with 3 convolutional layers + 2 fully connected layers to process 84×84 grayscale frames
- A **Replay Buffer** (capacity: 100,000) for experience replay to break temporal correlations
- Separate **policy network** and **target network**, with periodic target updates every 10,000 steps
- **Epsilon-greedy action selection** with exponential decay (ε: 1.0 → 0.01) for exploration
- Training with **Adam optimizer** (lr=1e-4), batch size 32, and discount factor γ=0.99

**✅ Conclusion:**
The DQN architecture successfully set up end-to-end training on a raw pixel-based Atari environment. The use of experience replay and a separate target network — the two key innovations of DQN — stabilize training by reducing correlation between consecutive updates and preventing oscillating Q-value targets. This experiment demonstrates how **deep learning combined with RL enables agents to learn directly from high-dimensional sensory inputs**, forming the foundation for modern deep RL systems.

---

> 📁 All experiments are implemented in Python (Google Colab) | Libraries: `NumPy` `OpenAI Gym` `PyTorch` `Matplotlib`
