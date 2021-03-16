import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
class Models:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self.actor = self.__get_actor()
        self.critic = self.__get_critic()
        #self.upper_bound

    def __get_actor(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-.00003, maxval=.00003)

        inputs = layers.Input(shape=(self.num_states,))
        out = layers.Dense(256, activation="relu", kernel_initializer=last_init)(inputs)
        out = layers.Dense(256, activation="relu", kernel_initializer=last_init)(out)
        out = layers.BatchNormalization()(out)
        outputs = layers.Dense(self.num_actions, activation="sigmoid", kernel_initializer=last_init)(out)

        # upplim_jthigh = 250*(np.pi/180)
        # lowlim_jthigh = 90*(np.pi/180)
        # diff_thigh = upplim_jthigh - lowlim_jthigh
        # upplim_jleg = 120*(np.pi/180)

        # #outputs = outputs * 1.57
        # mult = tf.constant([diff_thigh,upplim_jleg,diff_thigh,upplim_jleg,diff_thigh,upplim_jleg,diff_thigh,upplim_jleg,diff_thigh,upplim_jleg,diff_thigh,upplim_jleg,diff_thigh,upplim_jleg,diff_thigh,upplim_jleg,diff_thigh,upplim_jleg,diff_thigh,upplim_jleg])
        # add = tf.constant([lowlim_jthigh,0,lowlim_jthigh,0,lowlim_jthigh,0,lowlim_jthigh,0,lowlim_jthigh,0,lowlim_jthigh,0,lowlim_jthigh,0,lowlim_jthigh,0,lowlim_jthigh,0,lowlim_jthigh,0])
        # outputs = (outputs * mult) + add

        #outputs = outputs * 4

        model = tf.keras.Model(inputs, outputs)
        return model

    def __get_critic(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.0003, maxval=0.0003)

        # State as input
        state_input = layers.Input(shape=(self.num_states))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(self.num_actions))
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

