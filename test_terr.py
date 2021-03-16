import gym
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import model_from_json
from config import Settings
from policy import Policy
from reward import get_reward
import gym.envs.mujoco.mujoco_env as mujoco_env

from print_scores import print_theta #For nice plot
from print_scores import print_velocities #For nice plot
from print_scores import print_rewards #For nice plot

from models import Models
from nn import NeuralNetwork
import json

def get_action(obs, model):
    obs = obs.reshape(1,14)
    action = model(obs).numpy()[0]

    return action

num_eps_test = 10

#define settings
settings = Settings()

model = Models(14, 20)
actor = model.actor

mujoco_env.wedge_position_offset = 2.0
env = gym.make(settings.env_name)

# load weights into new model d
actor.load_weights("models/actor_terr.h5")
print("Loaded model actor_terr from disk")

for var in range(1,4):
    env.unwrapped.update_wedge(wedge_index=var)
    state = env.reset()

    distance_change = 0
    resetSettings = True

    on_slope = False
    for t in range(6000):
        env.render() 
        

        if on_slope:
            if state[0] < 2.9:
                grade = 0
            elif state[0] < 6.2:
                grade = -1 * var
            else:
                grade = 0

        else:

            if state[0] < 2.25:
                grade = 0
            elif state[0] < 5.55:
                grade = var
            elif state[0] < 8.85:
                grade = 0
            elif state[0] < 12.15:
                grade = -1 * var
            else:
                grade = 0



        obs = np.array(state[1:])
        obs = np.append(obs,grade)

        if t % 20 == 0:
            theta = get_action(obs, actor)

        kd=6
        pi = Policy(theta=theta, action_size=settings.action_size, action_min=settings.action_min, action_max=settings.action_max, kp=settings.control_kp, kd=kd, feq=settings.frequency, mode = "hzdrl", rs = resetSettings)
        action = pi.get_action(state)
        resetSettings = False

        # Perform action and recieve state and reward from environment.
        new_state, reward_params, done, _ = env.step(action)

        distance_change += (reward_params[1]-reward_params[2])

        if distance_change < 1 and t > 3000:
            #stuck
            done=True

        # Episode can finish either by reaching terminal state or max episode steps
        if done or t == 8000 - 1:
            print(t)
            break

        state = new_state
    print(distance_change)

env.close()