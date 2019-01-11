import os

import matplotlib
import numpy as np
import time
import tensorflow as tf

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, AveragePooling2D, Reshape, AveragePooling3D, Conv2D, Permute, \
    Convolution2D
from keras.optimizers import Adam
from keras.utils import plot_model

from matplotlib import pyplot as plt

from rl.agents.dqn import DQNAgent
from rl.callbacks import FileLogger
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy, MaxBoltzmannQPolicy
from rl.memory import SequentialMemory
from tensorflow.contrib.distributions.python.ops.bijectors import inline

from SeaGameEnv.sea_game import SeaGameEnv
from train_callback import EpisodeLogger

WEIGHT_FILE = 'weight.h5'
MODEL_FILE = 'model.json'
MODEL_HDF5 = 'model.h5'
#MODEL_HDF5 = 'model_pool8_dense256_dense128.h5'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
sess = tf.Session(config=config)
K.set_session(sess)

if __name__ == '__main__':
    env = SeaGameEnv(nb_npc=5, max_step=600,
                     ship_pool=4, tank_pool=8)
    nb_actions = env.action_space.n
    shape = (1, ) + env.observation_space.shape
    print(shape)

    model = Sequential()
    model.add(Flatten(input_shape=shape))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    plot_model(model, to_file='model.png')

    memory = SequentialMemory(limit=3000, window_length=1)

    policy = EpsGreedyQPolicy(eps=0.37)

    logger = EpisodeLogger()

    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=3000, policy=policy,
                   train_interval=3000, enable_double_dqn=True, gamma=0.6, target_model_update=0.01)
    dqn.compile(Adam(lr=0.001), metrics=['mae'])

    if os.path.exists(WEIGHT_FILE):
        dqn.load_weights(WEIGHT_FILE)

    history = dqn.fit(env, callbacks=[logger, FileLogger('log.json')], nb_steps=600*4000, visualize=False, verbose=2, log_interval=600)

    plt.plot(logger.rewards.values())
    plt.xlabel("episode")
    plt.ylabel("ep_reward")
    plt.show()

    dqn.model.save(MODEL_HDF5)

    env.is_test = True

    dqn.test(env, nb_episodes=5, visualize=True)

