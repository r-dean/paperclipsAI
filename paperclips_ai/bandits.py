"""
Learning algorthims which assume a constant state.
"""

import numpy as np

from paperclips_ai.adapter import WebAdapter
from paperclips_ai import State
from paperclips_ai.learner import Learner, ActionSelector

class eGreedyActionSelector(ActionSelector):    # pylint: disable=C0103:invalid-name
    """A selector of an e-greedy options"""
    def __init__(self, adapter: WebAdapter, state: State, action_ids: list[str], epsi: float):
        """Initializes the selector.

        Args:
            adapter: The adapter for the game.
            state: The state of the game.
            action_ids: The ids of all possible actions.
            epsi: The epsilon value for the e-greedy selection.
        """
        super().__init__(adapter, state, action_ids)
        self.epsi = epsi

    def choose(self) -> tuple[int, None]:
        """Choose and take an action.
        
        Returns:
            The action index and the value of the action.
        """
        # pick action if it hasnt been picked yet
        action = self.pick_unchosen()
        if action is not None:
            return action, None

        # otherwise pick based on upper confidence bound
        luck = np.random.rand()
        if luck < self.epsi:
            return np.random.randint(0, self.count), None

        return np.argmax(self.values), None

class ucbActionSelector(ActionSelector):        # pylint: disable=C0103:invalid-name
    """A selector of options based off upper confidence bounds."""
    def __init__(self, adapter: WebAdapter, state: State, action_ids: list[str], drift: float):
        """Initializes the selector.
        
        Args:
            adapter: The adapter for the game.
            action_ids: The ids of all possible actions.
            drift: The amount of drift to use in the ucb calculation.
        """
        super().__init__(adapter, state, action_ids)
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
        """Updates the weights and state."""
        if self.selector.actions_taken[self.action] <= 1:
            self.selector.values[self.action] = self.state.last_reward
        else:
            self.selector.values[self.action] += (
                1/self.selector.actions_taken[self.action]
                * (self.state.last_reward - self.selector.values[self.action]))

    def __repr__(self) -> str:
        """The string representation of the agents actions and policy."""
        info_str = super().__repr__()
        info_str += "Estimated Action Values:\n"
        for i in range(0, self.selector.count):
            info_str += f'\t{self.selector.action_els[i].name}: {self.selector.values[i]}\n'
        return info_str
