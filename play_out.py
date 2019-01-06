import os

import tensorflow as tf

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.utils import plot_model


from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

from SeaGameEnv.sea_game import SeaGameEnv, POOL
from train_callback import EpisodeLogger

WEIGHT_FILE = 'weight.h5f'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
sess = tf.Session(config=config)
K.set_session(sess)

env = SeaGameEnv(nb_npc=5, max_step=600)
nb_actions = env.action_space.n
shape = (1,) + (256//POOL, 512//POOL)
print(shape)
model = Sequential()
model.add(Flatten(input_shape=shape))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

plot_model(model, to_file='model.png')

memory = SequentialMemory(limit=3000, window_length=1)

policy = EpsGreedyQPolicy()

logger = EpisodeLogger()

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=3000, policy=policy,
               train_interval=3000, enable_double_dqn=True, gamma=.9, target_model_update=0.01,
               enable_dueling_network=True, dueling_type='avg')
dqn.compile(Adam(lr=0.001), metrics=['mae'])

if os.path.exists(WEIGHT_FILE):
    dqn.load_weights(WEIGHT_FILE)

env.is_test = True

dqn.test(env, nb_episodes=10, visualize=True)
dqn.save_weights(WEIGHT_FILE, True)