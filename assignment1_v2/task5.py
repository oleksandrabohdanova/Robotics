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

    def pick_action(self, obs):
        ### Using a policy pick an action for the state `obs`
        ### Використовуючи стратегію обрати дію для стану `obs`
        return np.argmax(self.policy[obs])


    def run(self):
        ### Using `self.transition_probs`, `self.states`, and `self.actions`, compute a policy.
        ### Викорстовуючи `self.transition_probs`, `self.states` та `self.actions`, обчисліть стратегію.
        
        ### Зверніть увагу: | Note:
        ### [(prob, next_state, reward, terminate), ...] = transition_probability[state][action]
        ### prob = probability(next_state | state, action)
        theta = 0.001
        discount = 0.95

        # policy evaluation
        def policy_evaluation(policy):
            value_function = np.zeros(len(self.states))
            return value_function
    
        # policy improvement
        while True:
            pass


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
    print(f"Average reward: {np.mean(rewards):.3f} (std={np.std(rewards):.3f})")

    env = gym.make(env_name, render_mode='human')
    reward = play_episode(env, policy_iteration, render=True, iter_max=50)


if __name__ == "__main__":
    print("Task 5.1 - Frozen Lake")
    task('FrozenLake-v1')

    print("Task 5.2 - Cliff Walking")
    task('CliffWalking-v1')
