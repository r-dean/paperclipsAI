import numpy as np

from paperclips_ai.learner import Learner, ActionSelector
from paperclips_ai.adapter import WebAdapter
from paperclips_ai.state import State

class SemiGradientSARSASelector(ActionSelector):
    """A selector that uses sarsa value function."""
    def __init__(self, adapter: WebAdapter, action_ids: list[str], state: State):
        super().__init__(adapter, action_ids)
        self.weights = np.zeros((self.count, state.size))
        self.step = 0
        self.state = state

    def choose(self):
        """Pick an action"""
        self.step += 1
        action = self.pick_unchosen()
        if action is not None:
            return action, self.value_function(self.state.vector, action, self.weights)

        values = np.array([
            (
                self.value_function(self.state.vector, action, self.weights)
                + 0.1 * (np.log(self.step) / self.actions_taken[action])
            ) for action in range(self.count)])
        action = np.argmax(values)
        return action, values[action]

    def value_function(self, state, action, weights):
        """Return estimated value of an action within a state"""
        return np.dot(weights[action], state)

    def get_gradient(self, state, action):
        """Get the gradient of taking an action"""
        return state[action]

class SemiGradientSARSALearner(Learner):
    """An agent that learns a linear function with SARSA algorithim"""
    def __init__(
            self,
            adapter: WebAdapter,
            selector: SemiGradientSARSASelector,
            stepsize: float = 0.1,
            gamma: float = 0.5):
        super().__init__(adapter, selector)
        self.stepsize = stepsize
        self.gamma = gamma

    def update(self):
        gradient  = self.selector.get_gradient(self.selector.state.vector, self.action)
        self.selector.weights += (
            self.stepsize
            * (self.last_reward + (self.gamma * self.exp_reward) - self.last_exp_reward)
            * gradient)
