import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from random import randint     
from print_scores import print_scores #For nice plot
from offline_data import choseOfflineAction, sigmoid

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
            actor_lr = 0.0000001
            critic_lr = 0.00005

        critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.buffer = Buffer(self.main_model, self.target_model, actor_optimizer, critic_optimizer)

        #Define default noise
        if refine_models:
            self.std_dev = 0.1
        else:
            self.std_dev = .5
        
        self.refine_models = refine_models


    def get_action(self, obs, noise):
        obs = obs.reshape(1,3)
        sampled_actions = self.main_model.actor(obs).numpy()[0]
        

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
            self.target_model.actor.set_weights(self.main_model.actor.get_weights())

            print("Revert")

        else:
            #Save actor
            if self.refine_models:
                num=""
            self.main_model.actor.save_weights("models/actor"+str(num)+".h5")
            
            print("Saved")
        self.main_model.critic.save_weights("models/critic.h5")

        #Save critic
        
            

    def train(self, env, total_episodes, pretrain, use_known_thetas, tau=0.001):
        steps=1000
        total_episodes=100
        if self.refine_models:
            total_episodes = 20

        previous_avg_reward = 540
        
        current_speed_list = []
        error = []
        timesteps = 0
        rewards=[]
    
        self.rewards = []
        for i in range(5):
            self.rewards.append(list())

        velocity_mean = 0.7
        velocity_std = 0.7

        reward_increment = 10

        for s in range(steps):
                
            for ep in range(total_episodes):

                if ep > total_episodes - 6:
                    noise = np.random.normal(0, 0, 20)
                    desired_velocity = ((ep - (total_episodes - 6)) * 0.2) + 0.4
                else:
                    if ep % 2 == 0:
                        noise = np.random.normal(0, self.std_dev, 20)
                    else:
                        noise = np.random.normal(0, 0.05, 20)
                    if ep % 40 == 0:
                        noise = np.random.normal(0, 0, 20)
                    desired_velocity = np.random.uniform(low=settings.desired_v_low,high=settings.desired_v_up)
   
                correct_theta = choseOfflineAction(desired_velocity)
                
                state = env.reset()
                if state is None:
                    state = np.zeros(settings.state_size)
                episodic_reward = 0
                distance_change = 0
                total_distance_change = 0

                total_current_speed_list = []
                resetSettings = True
                

                
                for t in range(settings.max_episode_length):
                    
                    #env.render()

                    #Update tuple
                    current_speed = state[7]         
                    current_speed_list += [current_speed]
                    while len(current_speed_list) > 200:
                        del current_speed_list[0]
                    current_speed_av = np.mean(np.array(current_speed_list))  
                    total_current_speed_list += [current_speed_av]              
                    error += [desired_velocity - current_speed]
                    while len(error) > 200:
                        del error[0]
                    obs = np.array([current_speed_av, 
                                desired_velocity,
                                state[1]
                                ])

                    obs = (obs - velocity_mean) / velocity_std

                    timesteps+=1
                                
                    if t % reward_increment == 0:    
                        if t > 0:                          
                            reward = get_reward(desired_velocity, np.mean(current_speed_list[-reward_increment:]), state[1], previous_height, False)
                   
                            self.buffer.record((init_obs, theta, reward, obs, correct_theta)) 
                            episodic_reward+=reward


                        if ep % 10 == 0 and use_known_thetas:
                            if ep % 30 ==0:
                                noise = np.random.normal(0, 0.0, 20) 
                            else:
                                noise = np.random.normal(0, 0.05, 20) 
                            theta = sigmoid(correct_theta) + noise
                            kd=6        

                        else:
                            theta = self.get_action(obs, noise)
                            kd=6

                        init_obs = obs
                        distance_change = 0

                        previous_height = state[1]                        
                    
                    pi = Policy(theta=theta, action_size=settings.action_size, action_min=settings.action_min, action_max=settings.action_max, kp=settings.control_kp, kd=kd, feq=settings.frequency, mode = "hzdrl", rs = resetSettings)
                    resetSettings = False

                    action = pi.get_action(state)

                    # Perform action and recieve state and reward from environment.
                    new_state, reward_params, done, _ = env.step(action)

                
                    distance_change += (reward_params[1]-reward_params[2])
                    total_distance_change += (reward_params[1]-reward_params[2])
                    #reward = get_reward(new_state, desired_velocity, current_speed_av)

                    if timesteps % 100 == 0 and ep < (total_episodes - 6):
                        #Update critic and actor using samples
                        self.buffer.learn(pretrain)
                        #Update target networks
                        self.__update_target(self.target_model.actor.variables, self.main_model.actor.variables, tau)
                        self.__update_target(self.target_model.critic.variables, self.main_model.critic.variables, tau)  



                    # End this episode when `done` is True
                    if done or t== settings.max_episode_length - 1:  
                        reward = get_reward(desired_velocity, np.mean(current_speed_list[-(t %reward_increment):]), state[1], previous_height, t < settings.max_episode_length - 1)
                                
                        self.buffer.record((init_obs, theta, reward, obs, correct_theta))  
                        episodic_reward+=reward

                        current_speed_list = []
                        error = []
                        break

                    state = new_state

                rewards.append(episodic_reward)
                print('episode ', total_episodes * s + ep, 'score %.2f' % episodic_reward, 'desired_velocity ',desired_velocity,'distance',total_distance_change,'t',t,'noise',np.mean(np.array(noise)),'trailing 100 games avg %.3f' % np.mean(rewards[-100:]))
                
                if ep >= total_episodes - 5:
                    log_string = ('episode ', total_episodes * s + ep, 'score %.2f' % episodic_reward, 'desired_velocity ',desired_velocity,'distance',total_distance_change,'noise',np.mean(np.array(noise)),'trailing 100 games avg %.3f' % np.mean(rewards[-100:]))
                    with open("train.txt", "a") as text_file:
                        text_file.write(str(log_string) + "\n")

            log_string = ('episode',(s+1) * total_episodes,'avg %.3f' % np.mean(rewards[-5:]))
            with open("train.txt", "a") as text_file:
                text_file.write(str(log_string) + "\n\n")

            avg_rewards = rewards[-5:]
            for i in range(5):
                self.rewards[i].append(avg_rewards[i])
            if previous_avg_reward > np.mean(avg_rewards):
                revert = True
            else:
                revert = False
                previous_avg_reward = np.mean(avg_rewards)
            
            

            if not self.refine_models:
                revert = False

            self.save_models((s+1) * total_episodes, revert)

            
            #if s > 20:
            print_scores(self.rewards, (s+1)*total_episodes, total_episodes, "DDPG Training - 10 Known Actions With Adjusted Reward")

        

    def handler(self, signal_received, frame):
        # Handle any cleanup here
        print('Saved_rewards')
        print_scores(self.rewards, 0, 100, "test")
        exit(0)


if __name__ == "__main__":

    load_models = True
    pretrain = False
    use_known_thetas = True
    refine_models = True

    #define settings
    settings = Settings()

    #create environment
    env = gym.make(settings.env_name)
    env.reset()

    #create DDPG object
    ddpg = DDPG_train(Policy, 3, 20, load_models, pretrain, refine_models)

    # Tell Python to run the handler() function when SIGINT is recieved
    signal(SIGINT, ddpg.handler)

    #train network
    ddpg.train(env, settings.num_episode, pretrain, use_known_thetas)




