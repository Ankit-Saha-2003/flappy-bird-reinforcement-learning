import time
import flappy_bird_gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env = flappy_bird_gym.make("FlappyBird-v0")
env.reset()

# Description of environment
print(f'Action space: {env.action_space}')
print(f'Observation space: {env.observation_space}')

# Examples of actions and states
print(f'Action space sample: {env.action_space.sample()}')
print(f'Observation space sample: {env.observation_space.sample()}')
## obs[0] = horizontal distance between agent and next pipe
## obs[1] = vertical distance between agent and next hole

# Example of result of action on state
print(f'Result sample: {env.step(env.action_space.sample())}')
## res[0] = new observation (hdist, vdist)
## res[1] = reward
## res[2] = whether episode is completed or not
## res[3] = diagnostic information useful for debugging

train_steps = 500000
episodes = 10

reward_per_episode = []
timesteps_per_episode = []

model = DQN('MlpPolicy', env, verbose=1, tensorboard_log='Logs/')
model.learn(total_timesteps=train_steps)
model.save('Models/DQN_Flappy_Bird')

evaluate_policy(model, env, n_eval_episodes=3, render=True)

for episode in range(episodes):
    obs = env.reset()
    done = False
    score = 0
    timesteps = 0

    while not done:
        action, states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward
        timesteps += 1
        env.render()
        time.sleep(1/30)

    reward_per_episode.append(score)
    timesteps_per_episode.append(timesteps)

print(reward_per_episode)
print(timesteps_per_episode)
env.close()