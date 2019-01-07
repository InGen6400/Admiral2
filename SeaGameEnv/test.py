from copy import deepcopy
from time import sleep

import numpy as np
from keras import Sequential
from keras.optimizers import Adam
from tensorflow.python.keras.models import model_from_json, load_model

from SeaGameEnv.sea_game import SeaGameEnv, ACTION_MEANS
from keras.models import Model

from learn import MODEL_FILE, WEIGHT_FILE, MODEL_HDF5


def process_state_batch(batch):
    batch = np.array(batch)
    return batch


def compute_batch_q_values(model, state_batch):
    batch = process_state_batch(state_batch)
    q_values = model.predict_on_batch(batch)
    assert q_values.shape == (len(state_batch), len(ACTION_MEANS))
    return q_values


model = load_model('../'+MODEL_HDF5)

env = SeaGameEnv(nb_npc=5, max_step=600)
obs = env.reset()
env.render()
while True:
    q_values = compute_batch_q_values(model, [[obs]]).flatten()
    action = np.argmax(q_values)
    obs, reward, done, _ = env.step(action)
    obs = deepcopy(obs)
    env.render()
    sleep(0.2)
    if done:
        env.reset()
