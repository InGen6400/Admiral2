import os
import warnings
from collections import deque
from typing import List, Any

from keras import Model
from keras.engine.training_utils import standardize_input_data
from keras.callbacks import Callback
from matplotlib import pyplot as plt
import keras.backend as K
import numpy as np
from rl.agents import DQNAgent


class EpisodeLogger(Callback):
    rewards: List[Any]

    def __init__(self, file_path, model_save_interval=1000):
        super().__init__()
        self.rewards = []
        self.mode_save_interval = model_save_interval
        self.model = None
        self.save_file = file_path

    def set_model(self, agent):
        self.model = agent.model

    def on_episode_end(self, episode, logs=None):
        self.rewards.append(logs['episode_reward'])
        if self.model and episode % self.mode_save_interval == 0:
            self.model.save(filepath=self.save_file+'/model'+str(episode)+'.h5')


class EarlyStopping(Callback):
    """Stop training when a monitored quantity has stopped improving.

    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline.
        restore_best_weights: whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
    """

    def __init__(self,
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 baseline=None,
                 restore_best_weights=False):
        super(EarlyStopping, self).__init__()

        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_episode = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.model = None

        self.monitor_op = np.greater
        self.min_delta *= 1
        self.best = -np.Inf
        self.hist = []
        self.current = []

    def set_model(self, agent):
        self.model = agent.model

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_episode = 0
        self.hist = []
        self.current = []
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_episode_end(self, episode, logs=None):
        current = logs['episode_reward']
        self.current.append(current)
        if current is None:
            return

        if episode % 100 == 0:
            if episode > 100:
                if self.monitor_op(sum(self.current)/len(self.current)-self.min_delta, sum(self.hist)/len(self.hist)):
                    self.wait = 0
                else:
                    self.wait += 1
                    self.hist = self.current
                    if self.wait >= self.patience:
                        self.stopped_episode = episode
                        self.model.stop_training = True
                self.current.clear()
            else:
                self.hist.append(current)

    def on_train_end(self, logs=None):
        if self.stopped_episode > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_episode + 1))
