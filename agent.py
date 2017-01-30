import gym
import universe  # register the universe environments
from universe import wrappers
import random
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import Adam

def ConvBlock(model, layers, filters):
    for i in range(layers):
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(filters, 3, 3, activation='relu'))
        model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

def create_action(observation_n):
    action_indexes = [random.randint(0, len(action_space) - 1) for ob in observation_n]  # TODO predict this value
    action_n = [action_space[indx] for indx in action_indexes]
    return action_n, action_indexes

def train_on_episode_state(state, model):
    model.fit([state[0], state[2]], state[1], len(state), nb_epoch=1)
    pass

def create_model(num_classes, size=(80,80)):
    model = Sequential()
    model.add(Lambda(input_shape=(3,)+size))
    ConvBlock(model, 1, 32)
    ConvBlock(model, 1, 64)
    ConvBlock(model, 1, 128)
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

env = gym.make('gym-core.PongDeterministic-v3')
env.configure(remotes=1)
env = wrappers.experimental.SafeActionSpace(env)
action_space = [[('KeyEvent', 'ArrowRight', True), ('KeyEvent', 'ArrowLeft', False)],  # Right
                [('KeyEvent', 'ArrowRight', False), ('KeyEvent', 'ArrowLeft', True)],  # Left
                [('KeyEvent', 'ArrowRight', False), ('KeyEvent', 'ArrowRight', False)]]  # stand still
model = create_model(num_classes=len(action_space))
observation_n = env.reset()

episode_state = []
#greed_epsilon = 0.1 Hvordan implemente i modellen?

#Test train on reward frames only? When hitting the ball.

while True:
    action_n, action_indexes = create_action(observation_n)
    observation_n, reward_n, done_n, info = env.step(action_n)
    episode_state.append((observation_n, reward_n, action_indexes))
    env.render()
    if done_n:
        train_on_episode_state(episode_state, model)
        episode_state = []

