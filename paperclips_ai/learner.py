"""
Base code for running learning algorithms.
"""

import logging
import time
import threading as th
from functools import wraps
import abc

import numpy as np

from paperclips_ai.adapter import WebAdapter, WebElement, EmptyWebElement
from paperclips_ai import State

class RepeatingThread(th.Thread):
    """A thread that runs its target function until cancelled."""
    def __init__(self, target, args=(), kwargs=None):
        """Initializes the thread."""
        super().__init__()
        self._stop_event = th.Event()
        self._target = target
        self._args = args
        self._kwargs = kwargs if kwargs is not None else {}

    def run(self):
        """Runs the threads target function until cancelled."""
        while not self._stop_event.is_set():
            self._target(*self._args, **self._kwargs)

    def stop(self):
        """Stops the thread from running its target function."""
        self._stop_event.set()


class ActionSelector(abc.ABC):
    """A set of availible actions for an agent to take."""
    def __init__(self, adapter: WebAdapter, state: State, action_ids: list[str]):
        """Get elements for actions.
        
        Args:
            adapter: The adapter for the game.
            state: The state of the game.
            action_ids: The ids of the actions.
        """
        self.logger = logging.getLogger(f'paperclips_ai.{self.__class__.__name__}')
        self.logger.setLevel(logging.DEBUG)
        self.adapter = adapter
        self.state = state
        self.action_els: list[WebElement] = (
            [EmptyWebElement('do nothing')]
            + [self.adapter.get_elem_by_id(action_id)
               for action_id in action_ids])
        self.count = len(action_ids) + 1
        self.values = np.zeros(self.count)
        self.actions_taken = np.zeros(self.count)

    @abc.abstractmethod
    def choose(self) -> tuple[int, float | None]:
        """Choose and take an action.
        
        Returns:
            The action index and the value of the action.
        """
        pass        # pylint:disable=W0107:unnecessary-pass

    def pick_unchosen(self) -> tuple[int | None, None]:
        """Take any unchosen action or return false."""
        untaken_actions = np.where(self.actions_taken==0)[0]
        if untaken_actions.size > 0:
            return untaken_actions[0]
        return None

    def take_action(self, i: int):
        """Take an action based on the provided index.
        
        Args:
            i: The index of the action to take.
        """
        self.actions_taken[i] += 1
        el: WebElement = self.action_els[i]
        el.click()

    def __len__(self):
        """The number of actions."""
        return self.count

    def __getitem__(self, i):
        """The value of an action at the given index."""
        return self.values[i]


class Learner(abc.ABC):
    """A base class for a learning algorithim that runs in the background until cancelled."""
    def __init__(self, adapter: WebAdapter, state: State, selector: ActionSelector):
        """Initializes a learner with a thread to do learning.
        
        Args:
            state: The state of the game.
            adapter: The adapter for the game.
            selector: The action selector for the agent.
        """
        self.thread: RepeatingThread = None
        self.state = state
        self.adapter = adapter
        self.selector: ActionSelector = selector
        self.logger = logging.getLogger(f'paperclips_ai.{self.__class__.__name__}')
        self.action = None
        self.next_action = None
        self.exp_reward = None
        self.last_exp_reward = None


    def stop(self):
        """Stops the current learning process."""
        if self.thread:
            self.thread.stop()
            while self.thread.is_alive():
                time.sleep(0.5)
            self.thread = None

    @abc.abstractmethod
    def update(self):
        """Update the weights."""
        pass    # pylint: disable=W0107:unnecessary-pass

    @staticmethod
    def learning_process(func):
        """Decorator for a learning process that should repeat until cancelled.
        
        Args:
            func: The function to be repeated.
        """
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.thread:
                self.stop()
            if kwargs.get('keep_running'):
                self.thread = RepeatingThread(target=func, args=(self, *args), kwargs=kwargs)
                self.thread.start()
            else:
                func(self, *args, **kwargs)

        return wrapper

    def __repr__(self):
        """Log estimated action values"""
        info_str = ""
        distribution = self.selector.actions_taken / np.sum(self.selector.actions_taken)
        info_str += "\nAction distribution:\n"
        for i in range(0, self.selector.count):
            info_str += f"\t{self.selector.action_els[i].name}: {distribution[i]}"
        info_str += "\n"

        return info_str

    @learning_process
    def learn(self, iterations=20, interval: float = 0.1, keep_running=False):  # pylint: disable=W0613
        """Learn a policy by interacting with the enviroment.
        
        Args:
            iterations: The number of iterations to run before checking if cancellation has occured.
            interval: The time to wait between each iteration.
        """
        self.next_action, self.exp_reward = self.selector.choose()
        for _ in range(iterations):
            self.action = self.next_action
            self.logger.debug("Chosen action: %s", self.action)
            prev_returns = self.state.returns
            self.selector.take_action(self.action)
            time.sleep(interval)
            self.state.last_reward = self.state.returns - prev_returns
            self.last_exp_reward = self.exp_reward
            self.next_action, self.exp_reward = self.selector.choose()
            self.update()
            self.state.update()

        # print results
        self.logger.info(repr(self))
