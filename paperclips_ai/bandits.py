"""
Learning algorthims which assume a constant state.
"""

import numpy as np

from paperclips_ai.adapter import WebAdapter
from paperclips_ai.learner import Learner, ActionSelector

class eGreedyActionSelector(ActionSelector):    # pylint: disable=C0103:invalid-name
    """A selector of an e-greedy options"""
    def __init__(self, adapter: WebAdapter, action_ids: list[str], epsi: float):
        super().__init__(adapter, action_ids)
        self.epsi = epsi

    def choose(self):
        """Choose and take an action."""
        # pick action if it hasnt been picked yet
        action = self.pick_unchosen()
        if action is not None:
            return action, None

        # otherwise pick based on upper confidence bound
        luck = np.random.rand()
        if luck < self.epsi:
            return self.take_action(np.random.randint(0, self.count))

        return np.argmax(self.values), None

class ucbActionSelector(ActionSelector):        # pylint: disable=C0103:invalid-name
    """A selector of options based off upper confidence bounds"""
    def __init__(self, adapter: WebAdapter, action_ids: list[str], drift: float):
        super().__init__(adapter, action_ids)
        self.drift = drift
        self.timestep = 0


    def choose(self):
        """Choose and take an action."""
        # increment the timestep
        self.timestep += 1
        # pick action if it hasnt been picked yet
        action = self.pick_unchosen()
        if action is not None:
            return action, None

        # otherwise pick based on upper confidence bound
        ucb_value = self.values + (
            self.drift * np.sqrt (np.log(self.timestep) / self.actions_taken))
        action = np.argmax(ucb_value)
        return action, None


class BanditLearner(Learner):
    """An RL agent which assumes a constant state."""
    def update(self):
        self.selector.values[self.action] += (
            1/self.selector.actions_taken[self.action]
            * (self.last_reward - self.selector.values[self.action]))
