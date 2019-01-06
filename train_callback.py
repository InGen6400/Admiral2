from typing import Dict, Any, List

from rl.callbacks import Callback


class EpisodeLogger(Callback):
    rewards: Dict[Any, Any]

    def __init__(self):
        super().__init__()
        self.rewards = {}

    def on_episode_end(self, episode, logs=None):
        self.rewards[episode] = logs['episode_reward']


