import gym
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import model_from_json
from config import Settings
from policy import Policy
from reward import get_reward
from offline_data import sigmoid
import pandas as pd

from print_scores import print_theta #For nice plot
from print_scores import print_velocities #For nice plot
from print_scores import print_rewards #For nice plot

from models import Models
from nn import NeuralNetwork
import json

def get_action(obs, model):
    obs = obs.reshape(1,3)
    action = model(obs).numpy()[0]

    return action

def load_model(filename):
    with open(filename) as f:    
    	data = json.load(f)
    print('loading file %s' % (filename))
    model_params = np.array(data[0]) # assuming other stuff is in data
    return model_params	

def theta_transform(theta):
    upplim_jthigh = 250*(np.pi/180)
    lowlim_jthigh = 90*(np.pi/180)
    upplim_jleg = 120*(np.pi/180)
    lowlim_jleg = 0

    thigh_const = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    leg_const = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    low_lim = (thigh_const*lowlim_jthigh) + (lowlim_jleg*leg_const)
    high_lim = (thigh_const*upplim_jthigh) + (upplim_jleg*leg_const)
    diff_lim = high_lim - low_lim

    #subtract lower limit
    theta = theta - low_lim
    #Divide by difference
    theta = theta / diff_lim
    #Perform inverse sigmoid
    theta = np.log(theta/(1-theta))
    #Return to tanh scale (-1,1)
    #theta = theta / 4
    return theta
num_eps_test = 10

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

model = Models(3, 20)
actor = model.actor

# load weights into new model
actor.load_weights("models/actor1.h5")
print("Loaded model from disk")

model = NeuralNetwork(input_dim=settings.conditions_dim,
                        output_dim=settings.theta_dim+1,
                        units=settings.nn_units,
                        activations=settings.nn_activations)
model_params = load_model('1900.json')
model.set_weights(model_params)

velocity_mean = 0.7
velocity_std = 0.7

thetas = []

# Start testing
timesteps = 0
total_episodes = 100
for j in range(total_episodes):

    for i in range(2):
        #i=1
        #i=0
        var=0

        state = env.reset()


        desired_velocity = np.random.uniform(low=1.0,high=settings.desired_v_up)
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
        episode_thetas = []

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
                obs = np.array([current_speed_av, 
                            desired_velocity,
                            state[1]])
                obs = (obs - velocity_mean) / velocity_std

            else:
                obs = np.array([current_speed_av, 
                            desired_velocity,
                            np.mean(np.array(error))])	
                


            init_obs = obs

            reset_Settings = False
            if i == 0:
                if var % 10 == 0:
                    # if var == 0:
                    #     reset_Settings = True
                    theta = get_action(obs, actor)
                    if var % 100 == 0:
                        episode_thetas.append(theta)
                    #theta = theta_transform(a_rightS)
                    #print(theta)
                    #theta = correct_theta / 4
                    kd=6
                    pi = Policy(theta=theta, action_size=settings.action_size, action_min=settings.action_min, action_max=settings.action_max, kp=settings.control_kp, kd=kd, feq=settings.frequency, mode = "hzdrl", rs = reset_Settings)
                action = pi.get_action(state)

            else:
                theta_kd = model.predict(obs)
                theta = sigmoid(theta_kd[0:settings.theta_dim])
                
                if var % 100 == 0:
                    episode_thetas.append(theta)
                # for i in range(20):
                #     thetas[i].append(theta[i])
                
                #kd = (1 / (1 + np.exp(-theta_kd[-1]))) * settings.control_kd
                kd=6
                pi = Policy(theta=theta, action_size=settings.action_size, action_min=settings.action_min, action_max=settings.action_max, kp=settings.control_kp, kd=kd, feq=settings.frequency, mode = "hzdrl")
                action = pi.get_action(state)

            
            # if var < 2:
            #     print(state)
            #     print(action)
                

            # Perform action and recieve state and reward from environment.
            new_state, reward_params, terminal, _ = env.step(action)

            distance_change += (reward_params[1]-reward_params[2])
            step += 1

            #reward = get_reward(new_state, desired_velocity, current_speed_av)
            test_arr.append(current_speed_av)

            #print(current_speed_av)

            #episodic_reward += reward
            var = var + 1

            # Episode can finish either by reaching terminal state or max episode steps
            if terminal or t == settings.max_episode_length - 1:

                current_speed_list = []
                error = []
                n=var
                break


            state = new_state
        if distance_change > 5:
            thetas.extend(episode_thetas)

        print(distance_change)


    
env.close()

df = pd.DataFrame(thetas)
df.to_csv("offline_terrain_data.csv")