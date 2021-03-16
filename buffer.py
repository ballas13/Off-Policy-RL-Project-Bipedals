import numpy as np
import tensorflow as tf
from print_scores import print_qvals

class Buffer:
    def __init__(self, main_model, target_model, actor_optimizer, critic_optimizer, gamma = 0.90, buffer_capacity=1000000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        self.gamma = gamma

        self.iterations = 1

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        #initialize models
        self.actor_model = main_model.actor
        self.critic_model = main_model.critic
        self.target_actor = target_model.actor
        self.target_critic = target_model.critic

        #initialize optimizers
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, main_model.num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, main_model.num_actions))
        self.correct_action_buffer = np.zeros((self.buffer_capacity, main_model.num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, main_model.num_states))

        self.q_values = []
        self.crit = []

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.correct_action_buffer[index] = obs_tuple[4]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    #@tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,pretrain, correct_actions
    ):
        #print(correct_actions)
        # Training and updating Actor & Critic networks.
        # Update critic
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            self.crit.append(tf.math.reduce_mean(y).numpy())
            # print("move",tf.reduce_mean(reward_batch)>tf.reduce_mean(.1*self.target_critic([next_state_batch, target_actions], training=True)))
            # print("reward",tf.reduce_mean(reward_batch))
            # print("Q-change",tf.reduce_mean(.1*self.target_critic([next_state_batch, target_actions], training=True)))


            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean((y - critic_value)**2)
            #print(critic_loss)

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        #Update actor
        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            if pretrain:
                diff_value = actions - correct_actions
                actor_loss = tf.math.reduce_mean(diff_value**2)
            else:
                critic_value = self.critic_model([state_batch, actions], training=True)     
                # Used `-value` as we want to maximize the value given
                # by the critic for our actions
                actor_loss = -tf.math.reduce_mean(critic_value)
                
                #actor_loss = tf.math.reduce_mean(critic_value)
                self.q_values.append(-actor_loss.numpy())
        #print(actor_loss)
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        # if self.iterations % 500 == 0:
        #     print(self.actor_model.trainable_variables[4])
            
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self, pretrain=False):
        if self.buffer_counter > 0:
            
            # Get sampling range
            record_range = min(self.buffer_counter, self.buffer_capacity)
            # Randomly sample indices
            batch_indices = np.random.choice(record_range, self.batch_size)

            # Convert to tensors
            state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
            action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices], dtype=tf.float32)
            correct_action_batch = tf.convert_to_tensor(self.correct_action_buffer[batch_indices], dtype=tf.float32)
            reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
            reward_batch = tf.cast(reward_batch, dtype=tf.float32)
            next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

            self.update(state_batch, action_batch, reward_batch, next_state_batch, pretrain, correct_action_batch)

            self.iterations+=1

            if self.iterations % 40000 == 0: 
                print_qvals(self.q_values, self.iterations)
                print_qvals(self.crit, self.iterations+1)
