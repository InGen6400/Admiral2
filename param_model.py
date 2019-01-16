import datetime
import os
import sys

from keras.callbacks import TensorBoard
from line_profiler import LineProfiler
import matplotlib
import numpy as np
import time
import tensorflow as tf
from hyperopt import Trials, tpe, STATUS_OK

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, AveragePooling2D, Reshape, AveragePooling3D, Conv2D, Permute, \
    Convolution2D
from keras.optimizers import Adam
from keras.utils import plot_model

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional


from rl.agents.dqn import DQNAgent
from rl.core import Agent
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy, MaxBoltzmannQPolicy
from rl.memory import SequentialMemory

from SeaGameEnv.sea_game import SeaGameEnv
from SeaGameEnv.ship_agent import ShipAgent
from train_callback import EpisodeLogger, EarlyStopping

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
sess = tf.Session(config=config)
K.set_session(sess)


# dummy data function
def data():
    return


def param_model():
    ship_pool = {{choice([4, 8, 16])}}
    tank_pool = {{choice([2, 4, 8])}}
    dense1 = {{choice([64, 128, 256])}}
    dense2 = {{choice([16, 32, 64, 128, 256])}}
    gamma = {{uniform(0, 1)}}
    eps = {{uniform(0, 1)}}

    env = SeaGameEnv(nb_npc=5, max_step=600,
                     ship_pool=ship_pool, tank_pool=tank_pool)
    env.seed(134613)
    nb_actions = env.action_space.n
    shape = (1, ) + env.observation_space.shape
    model = Sequential()
    model.add(Flatten(input_shape=shape))
    model.add(Dense(dense1))
    model.add(Activation('relu'))
    model.add(Dense(dense2))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    save_file = './saves/save_{}_{}_{}_{}_{:.0f}_{:.0f}'\
        .format(ship_pool, tank_pool, dense1, dense2, gamma * 1000, eps * 1000)
    os.makedirs(save_file, exist_ok=True)
    logger = [TensorBoard(log_dir='./logs/model_{}_{}_{}_{}_{:.0f}_{:.0f}'
                          .format(ship_pool, tank_pool, dense1, dense2, gamma*1000, eps*1000),
                          write_graph=False, write_images=True),
              EpisodeLogger(file_path=save_file, model_save_interval=500)]

    memory = SequentialMemory(limit=30000, window_length=1)
    policy = MaxBoltzmannQPolicy(eps=eps)

    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=3000, policy=policy,
                   train_interval=3000, enable_double_dqn=True, gamma=gamma, target_model_update=0.01)
    dqn.compile(Adam(lr=0.005), metrics=['mae'])

    rtn = dqn.fit(env, nb_steps=600*20000, visualize=False, verbose=2, callbacks=logger)

    rtn2 = dqn.test(env, nb_episodes=20, visualize=False)

    total_reward = np.sum(rtn2.history['episode_reward'])

    dqn.model.save('./saves/save_{}_{}_{}_{}_{:.0f}_{:.0f}/final_model.h5'
                   .format(ship_pool, tank_pool, dense1, dense2, gamma*1000, eps*1000))

    print(model.summary())
    print('ship_pool: ' + str(ship_pool))
    print('tank_pool: ' + str(tank_pool))
    print('dense_num: ' + str(dense1))
    print('dense_num: ' + str(dense2))
    print('eps' + str(eps))
    print('gamma' + str(gamma))
    print(datetime.datetime.today())
    print('\n\n\n')

    return {'loss': -total_reward, 'status': STATUS_OK, 'model': model}


def start_param():
    best_run, best_model = optim.minimize(model=param_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials(),
                                          eval_space=True)
    print("Best performing model chosen hyper-parameters:")
    print(best_run)


if __name__ == '__main__':
    start_param()
