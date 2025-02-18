import numpy as np

from paperclips_ai.learner import Learner, ActionSelector
from paperclips_ai.adapter import WebAdapter
from paperclips_ai.state import State

class SemiGradientSARSASelector(ActionSelector):
    """A selector that uses sarsa value function."""
    def __init__(self, adapter: WebAdapter, state: State, action_ids: list[str], drift: float = 0.1):
        """Initializes the selector.

        Args:
            adapter: The adapter for the game.
            state: The state of the game.
            action_ids: The ids of all possible actions.
        """
        super().__init__(adapter, state, action_ids)
        self.weights = np.zeros((self.count, state.size))
        self.step = 0
        self.state = state
        self.drift = drift

    def choose(self):
        """Pick an action."""
        self.step += 1
        action = self.pick_unchosen()
        if action is not None:
            return action, self.value_function(self.state.vector, action, self.weights)

        values = np.array([
            (
                self.value_function(self.state.vector, action, self.weights)
                + self.drift * (np.log(self.step) / self.actions_taken[action])
            ) for action in range(self.count)])
        action = np.argmax(values)
        return action, values[action]

    def value_function(self, state_vector, action, weights):
        """Return estimated value of an action within a state.
        
        Args:
            state: The state of the game.
            action: The action to take.
            weights: The weights for the value function.
        """
        return np.dot(weights[action], state_vector)

    def get_gradient(self, state_vector):
        """Get the gradient of taking an action.
        
        Args:
            state: The state of the game.
            action: The action to take.
        """
        return state_vector

class SemiGradientSARSALearner(Learner):
    """An agent that learns a linear function with SARSA algorithim."""
    def __init__(
            self,
            adapter: WebAdapter,
            state: State,
            selector: SemiGradientSARSASelector,
            stepsize: float = 0.1,
            gamma: float = 0.5):
        """Initializes the learner.
        
        Args:
            adapter: The adapter for the game.
            state: The state of the game.
            selector: The action selector for the agent.
            stepsize: The stepsize for the learner.
            gamma: The discount factor for future rewards.
        """
        super().__init__(adapter, state, selector)
        self.selector: SemiGradientSARSASelector = self.selector
        self.stepsize = stepsize
        self.gamma = gamma

    def update(self):
        """Update the weights of the agent."""
        gradient  = self.selector.get_gradient(self.selector.state.vector)
        self.selector.weights[self.action] += (
            self.stepsize
            * (self.state.last_reward + (self.gamma * self.exp_reward) - self.last_exp_reward)
            * gradient)

    def __repr__(self):
        """Log estimated action values"""
        info_str = super().__repr__()

        info_str += "\nWeights: \n"
        for i in range(self.selector.count):
            info_str += f'{self.selector.action_els[i].name}\n'
            for j in range(self.state.size):
                info_str += f'\t{self.state.features[j].name}: {self.selector.weights[i][j]}\n'
        info_str += "\n"

        info_str += "Estimated Action Values:\n"
        for i in range(self.selector.count):
            info_str += (
                f'\t{self.selector.action_els[i].name}: '
                f'{self.selector.value_function(self.state.vector, i, self.selector.weights)}')
        info_str += "\n"

        return info_str
