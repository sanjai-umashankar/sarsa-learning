# SARSA Learning Algorithm


## AIM
To develop a Python program for SARSA Algorithm

## PROBLEM STATEMENT
The objective is to train an optimal control policy in an unknown discrete Markov Decision Process (MDP) environment using model-free reinforcement learning, where an agent interacts with the environment step-by-step to maximize cumulative discounted returns. Because the underlying dynamics are initially unknown, the agent must balance exploration and exploitation using an $\epsilon$-greedy strategy combined with decaying learning rates ($\alpha$) and exploration rates ($\epsilon$) to iteratively learn the true action-value function $Q^*(s, a)$ via the SARSA algorithm.

## SARSA LEARNING ALGORITHM
1. **Initialization:**
   - Initialize the action-value table $Q(s, a)$ with zeros for all state-action pairs.
   - Allocate tracking arrays ($Q\_track$ and $pi\_track$) to record convergence over training episodes.

2. **Decay Schedules:**
   - Set up linear decay schedules for both the learning rate ($\alpha$) and the exploration rate ($\epsilon$) to ensure stable convergence over time.

3. **Episode Execution:**
   - **Reset** the environment to get the starting state $s$.
   - **Select** an initial action $a$ using an $\epsilon$-greedy policy derived from the current $Q$-table.

4. **Step-by-Step Updates (Inside Each Episode):**
   - Execute action $a$ in the environment to observe the next state ($s'$), reward ($r$), and termination status (`done`).
   - Select the next action ($a'$) from state $s'$ using the same $\epsilon$-greedy policy.
   - **Update the $Q$-value** using the SARSA update rule:
     $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') \cdot (1 - \text{done}) - Q(s, a) \right]$$
   - Transition to the next state-action pair: $s \leftarrow s'$ and $a \leftarrow a'$.

5. **Tracking & Outputs:**
   - Save the history of the $Q$-table and extract the greedy policy ($\pi$) after each episode.
   - Return the final optimal $Q$-function, state-value function $V(s) = \max_a Q(s, a)$, final greedy policy, and training history.

## SARSA LEARNING FUNCTION
### Name: SANJAI  U
### Register Number: 212224240145

```
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):
    
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    # Helper function to generate epsilon-greedy policy
    def select_action(state, epsilon):
        if np.random.random() < epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(Q[state])

    # Compute decay schedules
    alphas = np.linspace(init_alpha, min_alpha, int(n_episodes * alpha_decay_ratio))
    alphas = np.append(alphas, np.full(n_episodes - len(alphas), min_alpha))

    epsilons = np.linspace(init_epsilon, min_epsilon, int(n_episodes * epsilon_decay_ratio))
    epsilons = np.append(epsilons, np.full(n_episodes - len(epsilons), min_epsilon))

    for e in range(n_episodes):
        alpha = alphas[e]
        epsilon = epsilons[e]
        
        # Handle environment reset compatibility
        reset_res = env.reset()
        state = reset_res[0] if isinstance(reset_res, tuple) else reset_res
        
        action = select_action(state, epsilon)
        
        done = False
        while not done:
            # Handle env.step() returning either 4 or 5 values
            step_res = env.step(action)
            if len(step_res) == 5:
                next_state, reward, terminated, truncated, _ = step_res
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_res
            
            next_action = select_action(next_state, epsilon)
            
            # SARSA update rule
            td_target = reward + gamma * Q[next_state][next_action] * (not done)
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            
            state, action = next_state, next_action
            
        Q_track[e] = Q
        
        # Derive greedy policy for tracking
        current_policy = np.argmax(Q, axis=1)
        pi_track.append(current_policy)

    # Final derived policy and state-value function
    pi = np.argmax(Q, axis=1)
    V = np.max(Q, axis=1)
    
    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:
### Alpha and Epsilon schedules
<img width="916" height="530" alt="image" src="https://github.com/user-attachments/assets/f79bc550-0794-4779-9f4d-4bf2f2954cc7" />

### Optimal value functions and policy
<img width="932" height="716" alt="image" src="https://github.com/user-attachments/assets/c78a738b-d86f-442a-810f-4e05a1ccd5bc" />

### First-visit Monte-Carlo Control
<img width="926" height="937" alt="image" src="https://github.com/user-attachments/assets/0e07a253-87be-4fd4-b6d8-f626e2074635" />


### SARSA-Learning
<img width="945" height="961" alt="image" src="https://github.com/user-attachments/assets/914f2bfe-2dc6-4a94-b727-be5ba4380c7d" />

### First-visit Monte-Carlo vs SARSA-learning
<img width="1692" height="930" alt="ChatGPT Image Sep 4, 2026, 08_46_07 PM" src="https://github.com/user-attachments/assets/478ee8bc-2e59-4bf4-89bf-a55ab442fb08" />

## RESULT:

Thus, the Python program for SARSA implementation has been successfully executed.
