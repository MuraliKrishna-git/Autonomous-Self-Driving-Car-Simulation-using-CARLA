from stable_baselines3 import PPO
from steer_test import CarEnv

models_dir = "models//xxxxxxxxxx"   # enter your model directory name/number here

env = CarEnv()
env.reset()

model_path = f"{models_dir}//xxxxxx.zip"    # enter your model name/number here
model = PPO.load(model_path, env=env)

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        #env.render()
        #print('reward from current step: ',reward)