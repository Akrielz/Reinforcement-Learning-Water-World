from molecule import *
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import os

EPS = 0.0001
GAMMA = 0.90
# current_directory = os.getcwd() + "\\Trainings"
current_directory = "F:\\Programming Projects\\Python\\Neural Network Project\\Trainings"
agent_nr_location = "agent_nr.nr"

def get_models(nr_inputs, nr_actions, lr):
    input_shape = (nr_inputs,)
    nr_hidden = 256

    X_input = Input(input_shape)

    #X = Conv2D(32, 8, strides=(4, 4),padding="valid", activation="elu", data_format="channels_first", input_shape=input_shape)(X_input)
    #X = Conv2D(64, 4, strides=(2, 2),padding="valid", activation="elu", data_format="channels_first")(X)
    #X = Conv2D(64, 3, strides=(1, 1),padding="valid", activation="elu", data_format="channels_first")(X)
    #X = Flatten(input_shape=input_shape)(X_input)

    X = Dense(nr_hidden, activation="elu", kernel_initializer='he_uniform')(X_input)
    X = Dense(nr_hidden//2, activation="elu", kernel_initializer='he_uniform')(X)
    #X = Dense(256, activation="elu", kernel_initializer='he_uniform')(X)
    #X = Dense(64, activation="elu", kernel_initializer='he_uniform')(X)

    action = Dense(nr_actions, activation="softmax", kernel_initializer='he_uniform')(X)
    value = Dense(1, kernel_initializer='he_uniform')(X)

    Actor = Model(inputs = X_input, outputs = action)
    Actor.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=lr))

    Critic = Model(inputs = X_input, outputs = value)
    Critic.compile(loss='mse', optimizer=RMSprop(lr=lr))

    return Actor, Critic

class Neural_Network:
    def __init__(self):
        self.lr = 0.000025

        self.states, self.actions, self.rewards = [], [], []
        
        self.scores, self.averages = [], []
        self.score = 0.0
        self.max_average = -30000.0

        self.nr_inputs = NR_INPUTS
        self.action_size = NR_ACTIONS

        with open(current_directory + "//" + agent_nr_location, "r") as file:
            nr = int(file.readline())

        with open(current_directory + "//" + agent_nr_location, "w") as file:
            file.write(str(nr+1))

        self.Model_name = "Agent_" + str(nr)

        self.Actor, self.Critic = get_models(nr_inputs=self.nr_inputs, nr_actions=self.action_size, lr=self.lr)

    def remember(self, state, action, reward):
        self.states.append(state)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        self.actions.append(action_onehot)
        self.rewards.append(reward)

    def act(self, state):
        prediction = self.Actor.predict(state)[0]
        action = np.random.choice(self.action_size, p=prediction)
        return action

    def best_act(self, state):
        np_input_values = np.asarray([state])
        prediction = self.Actor.predict(np_input_values)[0]
        action = np.argmax(prediction)
        return action

    def discount_rewards(self, reward):
        # Compute the gamma-discounted rewards over an episode
        running_add = 0
        discounted_r = np.zeros_like(reward)
        # print(reward)
        for i in reversed(range(0,len(reward))):
            if reward[i] != 0:
                running_add = 0
            running_add = running_add * GAMMA + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) # normalizing the result
        # print(discounted_r)
        discounted_r /= np.std(discounted_r) + EPS # divide by standard deviation
        # print(discounted_r)
        return discounted_r

    def load(self, agent_id, to_compile=False):
        actor_name = "Agent_" + agent_id + "_Actor.h5"
        critic_name = "Agent_" + agent_id + "_Critic.h5"
        
        self.Actor = load_model(current_directory + "//" + actor_name, compile=to_compile)
        self.Critic = load_model(current_directory + "//" + critic_name, compile=to_compile)

    def save(self, save_type):
        self.Actor.save(current_directory + "//" + self.Model_name + "_" + save_type +'_Actor.h5')
        self.Critic.save(current_directory + "//" + self.Model_name + "_" + save_type + '_Critic.h5')

    def update(self, generation):
        self.scores.append(self.score)
        self.averages.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        average = self.averages[-1]

        self.save("last")
        if average >= self.max_average and len(self.scores[-50:]) == 50:
            self.max_average = average
            self.save("best")
            print("Generation[", generation, "]: Improvement Saved", sep="")

        # reshape memory to appropriate shape for training
        states = np.vstack(self.states)
        actions = np.vstack(self.actions)

        # Compute discounted rewards
        discounted_r = self.discount_rewards(self.rewards)

        # Get Critic network predictions
        values = self.Critic.predict(states)[:, 0]

        # Compute advantages
        advantages = discounted_r - values

        # training Actor and Critic networks
        self.Actor.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0)
        self.Critic.fit(states, discounted_r, epochs=1, verbose=0)

        # reset training memory
        self.states, self.actions, self.rewards = [], [], []
        self.score = 0

    def feed(self, input_values, eval_func, molecules, in_game_score):
        np_input_values = np.asarray([input_values])

        action = self.act(np_input_values)
        reward = eval_func(action, molecules, in_game_score)
        self.remember(input_values, action, reward)
        self.score += reward
        return action
