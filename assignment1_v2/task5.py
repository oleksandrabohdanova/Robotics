import numpy as np
import tqdm

try:
    import gymnasium as gym
except ImportError:
    print("Can't import gymansium. Tring to import gym...")
    import gym


# Action constants.
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

def play_episode(env, policy_iteration, render=False, iter_max=1000):
    done = False
    obs, _ = env.reset()
    total_reward = 0
    i_iter = 0
    while not done:
        action = policy_iteration.pick_action(obs)
        obs, rew, done, _, _ = env.step(action)
        if render:
            env.render()
        total_reward += rew

        if iter_max < i_iter:
            done = True
        i_iter += 1
    return total_reward


#######################################################
### Task: Implement Policy Iteration algorithm.     ###
### Завдання: Імплементувати ітерацію стратегіями   ###
#######################################################
class PolicyIteration:
    def __init__(self, transition_probs, states, actions):
        self.transition_probs = transition_probs
        self.states = states
        self.actions = actions
        self.policy = np.ones([len(self.states), len(self.actions)]) / len(self.actions)
        self.gamma = 0.99  
        self.epsilon = 0.0001  

    def pick_action(self, obs):
    
        return np.argmax(self.policy[obs])

    def compute_expected_return(self, state, action, value_table):
       
        expected = 0.0
        for prob, next_state, reward, done in self.transition_probs[state][action]:
          
            continuation = 0.0 if done else self.gamma * value_table[next_state]
            expected += prob * (reward + continuation)
        return expected

    def evaluate_policy(self):
        
        values = np.zeros(len(self.states))
        
        converged = False
        while not converged:
            max_change = 0.0
            
            for state in self.states:
                prev_value = values[state]
                
                # Calculate weighted sum over actions
                state_value = 0.0
                for action in self.actions:
                    action_probability = self.policy[state][action]
                    expected_return = self.compute_expected_return(state, action, values)
                    state_value += action_probability * expected_return
                
                values[state] = state_value
                max_change = max(max_change, abs(prev_value - values[state]))
            
            converged = (max_change < self.epsilon)
        
        return values

    def improve_policy(self, values):
        """Make policy greedy with respect to value function"""
        policy_changed = False
        
        for state in self.states:
            old_action = np.argmax(self.policy[state])
            
            # Calculate expected returns for each action
            action_returns = np.array([
                self.compute_expected_return(state, action, values)
                for action in self.actions
            ])
            
            # Select greedy action
            best_action = np.argmax(action_returns)
            
            # Update to deterministic greedy policy
            self.policy[state] = np.zeros(len(self.actions))
            self.policy[state][best_action] = 1.0
            
            if old_action != best_action:
                policy_changed = True
        
        return policy_changed

    def run(self):
        ### Using `self.transition_probs`, `self.states`, and `self.actions`, compute a policy.
        ### Викорстовуючи `self.transition_probs`, `self.states` та `self.actions`, обчисліть стратегію.
        
        ### Зверніть увагу: | Note:
        ### [(prob, next_state, reward, terminate), ...] = transition_probability[state][action]
        ### prob = probability(next_state | state, action)
        
        iterations = 0
        while True:
            # Evaluate current policy
            values = self.evaluate_policy()
            
            # Improve policy greedily
            changed = self.improve_policy(values)
            
            iterations += 1
            
            # Converged when policy stops changing
            if not changed:
                print(f"Converged after {iterations} policy improvements")
                break


def task(env_name):
    env = gym.make(env_name)
    transition_probability = env.unwrapped.P
    states = np.arange(env.unwrapped.observation_space.n)
    actions = [UP, RIGHT, DOWN, LEFT]
    policy_iteration = PolicyIteration(
        transition_probs=transition_probability,
        states=states,
        actions=actions
    )
    policy_iteration.run()
        
    rewards = []
    for _ in tqdm.tqdm(range(100)):
        reward = play_episode(env, policy_iteration)
        rewards.append(reward)
    success_rate = sum(1 for r in rewards if r > 0) / len(rewards) * 100
    print(f"Average reward: {np.mean(rewards):.3f} (std={np.std(rewards):.3f})")
    print(f"Success rate: {success_rate:.1f}%")

    env = gym.make(env_name, render_mode='human')
    reward = play_episode(env, policy_iteration, render=True, iter_max=50)


if __name__ == "__main__":
    print("Task 5.1 - Frozen Lake")
    task('FrozenLake-v1')

    print("Task 5.2 - Cliff Walking")
    task('CliffWalking-v1')