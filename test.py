import gym
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import model_from_json
from config import Settings
from policy import Policy

from reward import get_reward

from print_scores import print_theta #For nice plot
from print_scores import print_velocities #For nice plot
from print_scores import print_rewards #For nice plot
import offline_data

from models import Models
from nn import NeuralNetwork
import json

def get_action(obs, model):
    obs = obs.reshape(1,15)
    action = model(obs).numpy()[0]

    return action

def load_model(filename):
    with open(filename) as f:    
    	data = json.load(f)
    print('loading file %s' % (filename))
    model_params = np.array(data[0]) # assuming other stuff is in data
    return model_params	


#define settings
settings = Settings()

#create environment
env = gym.make(settings.env_name)
#env.render("human")
env.reset()

import logging
logging.basicConfig(filename='myapp.log', level=logging.INFO)
logging.info('Started')
logger = logging.Logger('catch_all')

model = Models(15, 20)
actor = model.actor

# load weights into new model
actor.load_weights("models/actor45200.h5")
print("Loaded model from disk")

model = NeuralNetwork(input_dim=settings.conditions_dim,
                        output_dim=settings.theta_dim+1,
                        units=settings.nn_units,
                        activations=settings.nn_activations)
model_params = load_model('online_method.json')
model.set_weights(model_params)

velocity_mean = 0.7
velocity_std = 0.7

    


desired_velocities = [0.6,0.8,1.0,1.2,1.4]

# Start testing
timesteps = 0

for test_ep in range(len(desired_velocities)):

    for i in range(2):
            

        state = env.reset()


        desired_velocity = desired_velocities[test_ep]
        step = 0
        ep_done = False
        episodic_reward = 0
        total_current_speed_list = []
        current_speed_list = []
        episode_velocities_error = []
        error = []
        episode_rewards = []
        distance_change = 0
        test_arr = []
        terminal = False
        reset_Settings=True

        for t in range(settings.max_episode_length):
            #env.render() 
            timesteps += 1

            #Update tuple
            current_speed = state[7]
            
            
            current_speed_list += [current_speed]
            
            while len(current_speed_list) > 200:
                del current_speed_list[0]
            current_speed_av = np.mean(np.array(current_speed_list))
            error += [desired_velocity - current_speed]
            while len(error) > 200:
                del error[0]
            total_current_speed_list += [current_speed_av]
            episode_velocities_error += [abs(desired_velocity - current_speed_av)]
            if i == 0:

                obs = np.array(state)
                obs = np.append(obs,desired_velocity)
            else:
                obs = np.array([current_speed_av, 
                            desired_velocity,
                            np.mean(np.array(error))])	

            init_obs = obs
            
            if i == 0:
                if t % 10 == 0:
                    if t == 0:
                        reset_Settings = True

                    theta = get_action(obs, actor)
                    kd=6
                pi = Policy(theta=theta, action_size=settings.action_size, action_min=settings.action_min, action_max=settings.action_max, kp=settings.control_kp, kd=kd, feq=settings.frequency, mode = "hzdrl", rs = reset_Settings)
                action = pi.get_action(state)

            else:
                if t == 0:
                    reset_Settings = True
                theta_kd = model.predict(obs)
                theta = theta_kd[0:settings.theta_dim]
                theta = offline_data.sigmoid(theta)
                kd=6
                pi = Policy(theta=theta, action_size=settings.action_size, action_min=settings.action_min, action_max=settings.action_max, kp=settings.control_kp, kd=kd, feq=settings.frequency, mode = "hzdrl", rs = reset_Settings)
                action = pi.get_action(state)

            reset_Settings = False

            # Perform action and recieve state and reward from environment.
            new_state, reward_params, terminal, _ = env.step(action)

            distance_change += (reward_params[1]-reward_params[2])
            step += 1

            test_arr.append(current_speed_av)


            # Episode can finish either by reaching terminal state or max episode steps
            if terminal or t == settings.max_episode_length - 1:
                current_speed_list = []
                error = []
                break


            state = new_state

        print(distance_change)

        if i==0:
            test = total_current_speed_list
            test1 = test_arr
        else:
            comparison = total_current_speed_list
            test2 = test_arr

    print_velocities(test1, test2, desired_velocity)
    
env.close()