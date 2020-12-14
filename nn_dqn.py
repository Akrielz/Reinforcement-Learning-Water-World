from molecule import *
from collections import deque
import tensorflow as tf
import numpy as np
import random
import math
import time
import glob
import io
import base64

from keras.models import load_model

num_features = NR_INPUTS
num_actions = NR_ACTIONS

# current_directory = os.getcwd() + "\\Trainings"
current_directory = "F:\\Programming Projects\\Python\\Neural Network Project\\Trainings"
agent_nr_location = "agent_nr.nr"

class DQN(tf.keras.Model):
    """Dense neural network class."""

    def __init__(self):
        super(DQN, self).__init__()

        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(128, activation="relu")
        self.dense3 = tf.keras.layers.Dense(
            num_actions, dtype=tf.float32
        )  # No activation

    def call(self, x):
        """Forward pass."""
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)


class ReplayBuffer(object):
    """Experience replay buffer that samples uniformly."""

    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, num_samples):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idx = np.random.choice(len(self.buffer), num_samples)

        for i in idx:
            elem = self.buffer[i]
            state, action, reward, next_state, done = elem
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.float32)

        return states, actions, rewards, next_states, dones


class Neural_Network_V2:
    def __init__(self):
        self.best = -300
        self.main_nn = DQN()
        self.target_nn = DQN()

        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        self.mse = tf.keras.losses.MeanSquaredError()

        self.epsilon = 1.0
        self.batch_size = 32
        self.discount = 0.99
        self.buffer = ReplayBuffer(100000)

        self.last_100_ep_rewards = []
        self.ep_reward = 0

        with open(current_directory + "\\" + agent_nr_location, "r") as file:
            nr = int(file.readline())

        with open(current_directory + "\\" + agent_nr_location, "w") as file:
            file.write(str(nr+1))

        self.model_name = "Agent_" + str(nr)

    def save(self, save_type):
        target_dir = current_directory + "\\QLearn\\"
        save_name = target_dir + self.model_name + "_" + save_type + "_"

        ws_main = self.main_nn.get_weights()
        for i, w in enumerate(ws_main):
            file_path = save_name + "main_weights_" + str(i)
            np.save(file_path, w)

        ws_target = self.target_nn.get_weights()
        for i, w in enumerate(ws_target):
            file_path = save_name + "target_weights_" + str(i)
            np.save(file_path, w)

    def load(self, agent_id, to_compile=False):
        target_dir = current_directory + "\\QLearn\\"
        save_name = target_dir + "Agent_" + agent_id + "_"

        ws_main = []
        ws_target = []
        for i in range(6):
            file_path_main = save_name + "main_weights_" + str(i) + ".npy"
            file_path_target = save_name + "target_weights_" + str(i) + ".npy"
            ws_main.append(np.load(file_path_main))
            ws_target.append(np.load(file_path_target))
        
        mock_state = [0 for x in range(85)]
        state_in = np.asarray([mock_state])

        p1 = self.main_nn(state_in)
        p2 = self.target_nn(state_in)

        self.target_nn.set_weights(ws_target)
        self.main_nn.set_weights(ws_main)


    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        """Perform a training iteration on a batch of data sampled from the experience
        replay buffer."""
        # Calculate targets.
        next_qs = self.target_nn(next_states)
        max_next_qs = tf.reduce_max(next_qs, axis=-1)
        target = rewards + (1.0 - dones) * self.discount * max_next_qs

        with tf.GradientTape() as tape:
            qs = self.main_nn(states)
            action_masks = tf.one_hot(actions, num_actions)
            masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)
            loss = self.mse(target, masked_qs)
            
        grads = tape.gradient(loss, self.main_nn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.main_nn.trainable_variables))

        return loss

    def select_epsilon_greedy_action(self, state, epsilon):
        """Take random action with probability epsilon, else take best action."""
        result = np.random.random_sample()
        if result < epsilon:
            return np.random.randint(num_actions, size=1)[0]  # Random action (left or right).
        else:
            return np.argmax(self.main_nn(state)[0])
            #return tf.keras.backend.get_value(tf.math.argmax(self.main_nn(state)[0]))
            #return tf.arg_max(self.main_nn(state)[0])  # Greedy action for state.

    def act(self, state):
        state_in = np.asarray([state])
        action = self.select_epsilon_greedy_action(state_in, self.epsilon)
        return action

    def best_act(self, state):
        state_in = np.asarray([state])
        action = self.select_epsilon_greedy_action(state_in, 0)
        return action

    def feed(self, state, action, next_state, reward, done, cur_frame):
        self.ep_reward += reward
        # Save to experience replay.
        self.buffer.add(state, action, reward, next_state, done)

        if (cur_frame+1) % 250 == 0:
            self.target_nn.set_weights(self.main_nn.get_weights())

        if len(self.buffer) >= self.batch_size:
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            loss = self.train_step(states, actions, rewards, next_states, dones)

    def update(self, generation):
        if generation < 950:
            self.epsilon -= 0.001

        if len(self.last_100_ep_rewards) == 100:
            self.last_100_ep_rewards = self.last_100_ep_rewards[1:]
        self.last_100_ep_rewards.append(self.ep_reward)

        mean = np.mean(self.last_100_ep_rewards)

        if generation % 50 == 0:
            print("Generation[", generation, "]: ",
            "\n |  Epsilon: ", self.epsilon, 
            "\n |_ Reward in last 100 Generations: ", mean,
            sep="")

        self.ep_reward = 0
        self.save("last")
        if generation > 50 and self.best < mean:
            self.best = mean
            print("Generation[", generation, "]: Improvement saved!", sep="")
            self.save("best")
