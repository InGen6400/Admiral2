import os
from threading import Thread

import tensorflow as tf

from keras import backend as K
from keras.callbacks import TensorBoard
from keras.engine.saving import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.utils import plot_model


from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy, MaxBoltzmannQPolicy
from rl.memory import SequentialMemory

from SeaGameEnv.sea_game import SeaGameEnv
from train_callback import EpisodeLogger, EarlyStopping

WEIGHT_FILE = 'weight.h5'
MODEL_FILE = 'model.json'
MODEL_HDF5 = 'model.h5'
# MODEL_HDF5 = 'model_pool8_dense256_dense128.h5'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
sess = tf.Session(config=config)
K.set_session(sess)


def learn_episode(episode, ship_pool, tank_pool, dense_unit1=0, dense_unit2=0, dense_unit3=0, loggers=None, model=None):
    env = SeaGameEnv(nb_npc=1, max_step=400,
                     ship_pool=ship_pool, tank_pool=tank_pool)
    nb_actions = env.action_space.n
    shape = (1, ) + env.observation_space.shape
    print(shape)

    if model is None:
        print('Create Model')
        model = Sequential()
        model.add(Flatten(input_shape=shape))
        if dense_unit1 != 0:
            model.add(Dense(dense_unit1))
            model.add(Activation('relu'))
        if dense_unit2 != 0:
            model.add(Dense(dense_unit2))
            model.add(Activation('relu'))
        if dense_unit3 != 0:
            model.add(Dense(dense_unit3))
            model.add(Activation('relu'))
        model.add(Dense(nb_actions))
        model.add(Activation('linear'))
        print(model.summary())
        plot_model(model, to_file='model.png')

    memory = SequentialMemory(limit=30000, window_length=1)

    policy = MaxBoltzmannQPolicy()

    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=3000, policy=policy,
                   train_interval=3000, enable_double_dqn=True, gamma=0.75, target_model_update=0.01)
    dqn.compile(Adam(lr=0.0005), metrics=['mae'])

    history = dqn.fit(env, callbacks=loggers, nb_steps=600*episode, visualize=False, verbose=2, log_interval=600)

    return dqn.model


if __name__ == '__main__':
    episode = 100000

    ship_pool = 4
    tank_pool = 4
    dense1 = 256
    dense2 = 128
    dense3 = 64

    save_file = './saves/save-2agent_{}_{}_{}_{}_{}_lr5'\
        .format(ship_pool, tank_pool, dense1, dense2, dense3)
    os.makedirs(save_file, exist_ok=True)
    logger = [TensorBoard(log_dir='./logs/model-2agent-lr5_'
                                  + str(ship_pool)+'_'
                                  + str(tank_pool)+'_'
                                  + str(dense1)+'_'
                                  + str(dense2)+'_'
                                  + str(dense3),
                          write_graph=False, write_images=True),
              EpisodeLogger(file_path=save_file, model_save_interval=1000)]

    '''
    if os.path.exists(MODEL_HDF5):
        model = load_model(MODEL_HDF5)
        print('loaded model')
    else:
    '''
    model = None

    learn_episode(episode, ship_pool, tank_pool, dense_unit1=dense1, dense_unit2=dense2, dense_unit3=dense3, loggers=logger, model=model)

    if model is not None:
        model.save('learned_model'+str(episode)+'.h5')



