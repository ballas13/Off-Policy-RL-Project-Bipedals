import gym
import gym.envs.mujoco.mujoco_env as mujoco_env
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from random import randint     
from print_scores import print_scores, print_terrain_scores #For nice plot
from offline_data import correct_theta_14, sigmoid
import math
import pandas as pd

from policy import Policy
from tensorflow.keras.models import model_from_json

from signal import signal, SIGINT
from sys import exit

from time import time

from print_scores import print_velocities1
from models import Models
from nn import NeuralNetwork
import json

import warnings
warnings.filterwarnings("ignore")

#Model specific libraries
from models import Models
from buffer import Buffer
from config import Settings
from policy import Policy
from reward import get_reward

class DDPG_train:
    def __init__(self, policy, num_states, num_actions, load_models, pretrain, refine_models):
        #get main actor and critic models
        self.main_model = Models(num_states, num_actions)
        #get target models
        self.target_model = Models(num_states, num_actions)
        if load_models:
            #Load actor
            self.main_model.actor.load_weights("models/actor.h5")
            print("Loaded actor model from disk")

            #Load Critic
            self.main_model.critic.load_weights("models/critic.h5")
            print("Loaded critic model from disk")

        self.target_model.actor.set_weights(self.main_model.actor.get_weights())
        self.target_model.critic.set_weights(self.main_model.critic.get_weights())


        #define default buffer
        critic_lr = 0.0001
        if pretrain:
            actor_lr = 0.000005
        else:
            actor_lr = 0.000005

        if refine_models:
            actor_lr = 0.000005
            critic_lr = 0.0001

        critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.buffer = Buffer(self.main_model, self.target_model, actor_optimizer, critic_optimizer)

        #Define default noise
        if refine_models:
            self.std_dev = 0.05
        else:
            self.std_dev = 0.2
        
        self.refine_models = refine_models
        self.num_states = num_states

    def theta_transform(self, theta):
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
        theta = theta / 4
        return theta


    def get_action(self, obs, noise):
        obs = obs.reshape(1,self.num_states)
        sampled_actions = self.main_model.actor(obs).numpy()[0]

        #Uncomment to see input to sigmoid layer
        # if randint(1, 400) == 8:
        #     model_output = self.main_model.actor.get_layer(index=4).output
        #     m = tf.keras.Model(inputs=self.main_model.actor.input, outputs=model_output)
        #     out = m(obs).numpy()[0]
        #     print(np.log(out/(1-out)))
        

        # Adding noise to action
        theta = sampled_actions + noise

        return theta


    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def __update_target(self,target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def save_models(self, num, revert):
        #num = ""
        if revert:
            #Load actor
            self.main_model.actor.load_weights("models/actor.h5")
            print("Loaded actor model from disk")
            self.target_model.actor.set_weights(self.main_model.actor.get_weights())

        else:
            #Save actor
            if self.refine_models:
                num=""
            self.main_model.actor.save_weights("models/actor"+str(num)+".h5")
            print("Saved model to disk")

        #Save critic
        self.main_model.critic.save_weights("models/critic.h5")
        print("Saved model to disk")
            

    def train(self, env, total_episodes, pretrain, use_known_thetas, tau=0.001):
        steps=1000
        total_episodes=100
        # if self.refine_models:
        #     total_episodes = 20

        offline_data = pd.read_csv('offline_terrain_data.csv').to_numpy()
        offline_data = np.delete(offline_data, 0, 1)
        
        timesteps = 0
        rewards=[]
    
        self.rewards = []
        for i in range(3):
            self.rewards.append(list())


        reward_increment = 20
        total_rewards=[]
        mujoco_env.wedge_position_offset = 2.0
        env = gym.make(settings.env_name)         
        on_slope=False
        max_steps = 10000

        for s in range(steps):
            mujoco_env.wedge_position_offset = -4.0
            env = gym.make(settings.env_name)
            on_slope=True
            for ep in range(total_episodes):
                if ep % 2 == 0 or ep>=total_episodes-3:
                    noise = np.random.normal(0, 0, 20)
                else:
                    noise = np.random.normal(0, self.std_dev, 20)

                if ep>=total_episodes-3 or randint(1,2)==1:
                    useCurrentPolicy = True
                else:
                    useCurrentPolicy = False

                correct_theta = offline_data[randint(0,len(offline_data)-1),:]
                if ep == 20:
                    mujoco_env.wedge_position_offset = 2.0
                    env = gym.make(settings.env_name)         
                    on_slope=False

                if ep>=total_episodes-3:
                    slope = ep-(total_episodes-4)
                    max_steps = 8000
                else:
                    slope = randint(1,3)

                    max_steps = 6000
               

                env.unwrapped.update_wedge(wedge_index=slope)
                if on_slope:
                    state = env.reset(slope)
                else:
                    state = env.reset()
                
                if state is None:
                    state = np.zeros(settings.state_size)
                episodic_reward = 0
                distance_change = 0
                total_distance_change = 0

                resetSettings = True

                doneInit = False
                episode_rewards = []
                current_speed_list=[]
                
                for t in range(max_steps):
                    #env.render()
                    #Update tuple
                    current_speed = state[7]         
                    current_speed_list += [current_speed]
                    while len(current_speed_list) > 200:
                        del current_speed_list[0]
                    current_speed_av = np.mean(np.array(current_speed_list))  

                    if len(total_rewards) > 0: 
                        self.buffer.record(total_rewards.pop())

                    if on_slope:
                        if state[0] < 2.9:
                            grade = 0
                        elif state[0] < 6.2:
                            grade = -1 * slope
                        else:
                            grade = 0

                    else:

                        if state[0] < 2.25:
                            grade = 0
                        elif state[0] < 5.55:
                            grade = slope
                        elif state[0] < 8.85:
                            grade = 0
                        elif state[0] < 12.15:
                            grade = -1 * slope
                        else:
                            grade = 0



                    obs = np.array(state[1:])
                    obs = np.append(obs,grade)

                    timesteps+=1
                                
                    if t % reward_increment == 0:    
                        if doneInit:                          
                            reward = distance_change * 5
                            if abs(state[2] - 0.18) < 0.2:
                                reward += 0.5
                            if current_speed_av < 0.5 or current_speed_av > 2.0:
                                reward -= 0.5

                            episode_rewards.append((init_obs, theta, reward, obs, correct_theta)) 
                        else:
                            total_distance_change = 0


                        if (not useCurrentPolicy) and use_known_thetas:
                            if ep % 2 == 0:
                                noise = np.random.normal(0, 0, 20)
                            else:
                                noise = np.random.normal(0, self.std_dev, 20)
                            theta = correct_theta + noise
                            kd=6        

                        else:
                            
                            theta = self.get_action(obs, noise)
                            kd=6

                        init_obs = obs
                        distance_change = 0
                        doneInit=True
                                        
                    
                    pi = Policy(theta=theta, action_size=settings.action_size, action_min=settings.action_min, action_max=settings.action_max, kp=settings.control_kp, kd=kd, feq=settings.frequency, mode = "hzdrl", rs = resetSettings)
                    resetSettings = False

                    action = pi.get_action(state)

                    # Perform action and recieve state and reward from environment.
                    new_state, reward_params, done, _ = env.step(action)
        
                    distance_change += (reward_params[1]-reward_params[2])

                    if total_distance_change < 1 and t > 3000:
                        #stuck
                        done=True
                    
                    total_distance_change += (reward_params[1]-reward_params[2])
                    #print(total_distance_change)
                    #print(total_distance_change)

                    if timesteps % 100 == 0 and  ep < (total_episodes - 3):
                        #Update critic and actor using samples
                        self.buffer.learn(pretrain)
                        #Update target networks
                        self.__update_target(self.target_model.actor.variables, self.main_model.actor.variables, tau)
                        self.__update_target(self.target_model.critic.variables, self.main_model.critic.variables, tau)  



                    # End this episode when `done` is True
                    if done or t== max_steps - 1:  
                        if t < max_steps - 1:
                            reward = -1
                            episode_rewards.append((init_obs, theta, reward, obs, correct_theta))  
                        adj_rewards=[]
                        if total_distance_change > 15:
                            adj_total_distance = 15
                        else:
                            adj_total_distance = total_distance_change
                        rew_val = 0
                        for val in episode_rewards:
                            listVal = list(val)
                            listVal[2] += ((adj_total_distance / 6) * (t / max_steps))
                            episodic_reward += listVal[2]
                            adj_rewards.append(tuple(listVal))
                            rew_val+=1
                        if ep<total_episodes - 3:
                            total_rewards.extend(adj_rewards)    
                        break

                    state = new_state

                rewards.append(episodic_reward)
                print('episode ', total_episodes * s + ep, 'score %.2f' % episodic_reward, 'distance',total_distance_change,'t',t,'noise',np.mean(np.array(noise)),'Policy', useCurrentPolicy, 'Slope',slope)
                
                if ep >= total_episodes - 3:
                    log_string = ('episode ', total_episodes * s + ep, 'score %.2f' % episodic_reward, 'distance',total_distance_change,'t',t,'noise',np.mean(np.array(noise)),'trailing 100 games avg %.3f' % np.mean(rewards[-100:]))
                    with open("train.txt", "a") as text_file:
                        text_file.write(str(log_string) + "\n")

            log_string = ('episode',(s+1) * total_episodes,'avg %.3f' % np.mean(rewards[-3:]))
            with open("train.txt", "a") as text_file:
                text_file.write(str(log_string) + "\n\n")

            avg_rewards = rewards[-3:]
            for i in range(3):
                self.rewards[i].append(avg_rewards[i])

            self.save_models((s+1) * total_episodes, False)


            print_terrain_scores(self.rewards, (s+1)*total_episodes, total_episodes, "DDPG Training Terrain with Known Actions")

        

    def handler(self, signal_received, frame):
        # Handle any cleanup here
        print('Saved_rewards')
        print_terrain_scores(self.rewards, 0, 100, "DDPG Training Terrain with Known Actions")
        exit(0)


if __name__ == "__main__":

    load_models = False
    pretrain = False
    use_known_thetas = True
    refine_models = False

    #define settings
    settings = Settings()

    #create environment
    env = gym.make(settings.env_name)
    env.unwrapped.update_wedge(wedge_index=1)

    env.reset()

    #create DDPG objectx
    ddpg = DDPG_train(Policy, 14, 20, load_models, pretrain, refine_models)

    # Tell Python to run the handler() function when SIGINT is recieved
    signal(SIGINT, ddpg.handler)

    #train network
    ddpg.train(env, settings.num_episode, pretrain, use_known_thetas)




