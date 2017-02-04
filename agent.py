import gym
import universe  # register the universe environments
from universe import wrappers
import random
from collections import deque
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import Adam
import numpy as np

NUM_FRAMES_STATE = 4  # Current impl of how many frames to remember state from.
REPLAY_MEMORY = 10  # Number of replays to store in memory for Experience Replay to randomly sample from.

def ConvBlock(model, layers, filters):
    for i in range(layers):
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(filters, 3, 3, activation='relu'))
        model.add(BatchNormalization())
    #model.add(MaxPooling2D((2, 2), strides=(2, 2))) # We care about positioning of the ball, and don't want to discard it. Ref nervanasys


def create_action(observation_n):
    action_indexes = [random.randint(0, len(action_space) - 1) for ob in observation_n]  # TODO predict this value
    action_n = [action_space[indx] for indx in action_indexes]
    return action_n, action_indexes


# Rewards for all observations in an episode.
# TODO how do we sample the reward given at a single frame if randomly sampled from the pool in Experience Replay.
def discount_reward(rewards):
    discount = 0.95
    total_rewards = sum(rewards)
    discounted_rewards = []
    for i in range(rewards):
        discounted_rewards.append(total_rewards)
        total_rewards *= discount # progressively reduce reward for each action in the future.
    return discounted_rewards

def train_on_episode_state(state, model):
    rewards = state[2]
    observations = state[0]
    for index in range(NUM_FRAMES_STATE):
        mean_reward = sum(rewards[index:index+NUM_FRAMES_STATE])/NUM_FRAMES_STATE
        model.fit(observations[index:index+NUM_FRAMES_STATE])

        #TODO train on current memory-state
    model.fit([state[0], state[2]], state[1], len(state), nb_epoch=1)
    pass

def predict_reward_of_state(state):
    #predict up and down scores
    #predict up and down scores for following N frames given the highest score action of each.
        #So fan out in all directions and search for the optimal branch?
    return 0

def update_n_dim_observation(buffer, observation):
    buffer.appendleft(observation)
    buffer.pop()  # TODO ensure this pops right side, and appends left-side
    return np.array(list(buffer))

def create_model(num_classes, size=(80,80)):
    model = Sequential()
    model.add(Lambda(input_shape=(NUM_FRAMES_STATE,) + size))
    ConvBlock(model, 1, 32)
    ConvBlock(model, 1, 64)
    ConvBlock(model, 1, 64)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

env = gym.make('gym-core.PongDeterministic-v3')
env.configure(remotes=1)
env = wrappers.experimental.SafeActionSpace(env)
action_space = [[('KeyEvent', 'ArrowRight', True), ('KeyEvent', 'ArrowLeft', False)],  # Right
                [('KeyEvent', 'ArrowRight', False), ('KeyEvent', 'ArrowLeft', True)]]  # Left
                #[('KeyEvent', 'ArrowRight', False), ('KeyEvent', 'ArrowRight', False)]]  # stand still
model = create_model(num_classes=len(action_space))
observation_n = env.reset()

episode_state = []
#greed_epsilon = 0.1 Hvordan implemente i modellen?

#Test train on reward frames only? When hitting the ball.
# Compress 4 frames together for state first? Add RNN to keep state later. Pass frame to CNN & RNN, merge output in functional API


## INit with 4 of the first frame.
def init_n_dim_observation(observation_n):
    n_dimensional_observation = deque()
    for n in range(NUM_FRAMES_STATE):
        n_dimensional_observation.appendleft(observation_n)
    return n_dimensional_observation

n_dimensional_observation = init_n_dim_observation(observation_n)

while True:
    action_n, action_indexes = create_action(observation_n)
    observation_n, reward_n, done_n, info = env.step(action_n)
    obs_n_with_movement = update_n_dim_observation(n_dimensional_observation, observation_n)
    episode_state.append((observation_n, reward_n, action_indexes))
    env.render()
    if done_n:
        train_on_episode_state(episode_state, model)
        episode_state = []



# Given this collection of frames and action X, predict estimated discounted Reward.

#  for action in action_space: predict(data, action) ## result = discounted reward.
## np.argmax(results) == action to take, which predicts highest amount of future reward.
