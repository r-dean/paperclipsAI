"""
Base code for running learning algorithms.
"""

import logging
import time
import threading as th
from functools import wraps
import abc

import numpy as np

from paperclips_ai.adapter import WebAdapter, WebElement

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
    def __init__(self, adapter: WebAdapter, action_ids: list[str]):
        """Get elements for actions."""
        self.logger = logging.getLogger(f'paperclips_ai.{self.__class__.__name__}')
        self.logger.setLevel(logging.DEBUG)
        self.adapter = adapter
        self.action_els: list[WebElement] = [
            self.adapter.get_elem_by_id(action_id) for action_id in action_ids]
        self.action_els.insert(0, None)
        self.count = len(action_ids) + 1
        self.values = np.zeros(self.count)
        self.actions_taken = np.zeros(self.count)

    @abc.abstractmethod
    def choose(self):
        """Choose and take an action."""
        pass        # pylint:disable=W0107:unnecessary-pass

    def pick_unchosen(self):
        """Take any unchosen action or return false."""
        untaken_actions = np.where(self.actions_taken==0)[0]
        if untaken_actions.size > 0:
            return untaken_actions[0]
        return None

    def take_action(self, i):
        """Take an action based on the provided index."""
        self.actions_taken[i] += 1
        if i == 0:
            return 0
        el: WebElement = self.action_els[i]
        el.click()
        return i

    def log_values(self):
        """Log estimated action values"""
        info_str = "Estimated Action Values:\n"
        info_str += f"None: {self.values[0]}"
        for i in range(1, self.count):
            info_str += f"\t{self.action_els[i].name}: {self.values[i]}"

        distribution = self.actions_taken / np.sum(self.actions_taken)
        info_str += "\nAction distribution:\n"
        info_str += f"None: {distribution[0]}"
        for i in range(1, self.count):
            info_str += f"\t{self.action_els[i].name}: {distribution[i]}"
        info_str += "\n"

        self.logger.info(info_str)

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        return self.values[i]


class Learner(abc.ABC):
    """A base class for a learning algorithim that runs in the background until cancelled."""
    def __init__(self, adapter: WebAdapter, selector: ActionSelector):
        """Initializes a learner with a thread to do learning."""
        self.thread: RepeatingThread = None
        self.adapter = adapter
        self.selector: ActionSelector = selector
        self.logger = logging.getLogger(f'paperclips_ai.{self.__class__.__name__}')
        self.action = None
        self.next_action = None
        self.last_reward = None
        self.exp_reward = None
        self.last_exp_reward = None


    def stop(self):
        """Stops the current learning process."""
        self.thread.stop()
        while self.thread.is_alive():
            time.sleep(0.5)
        self.thread = None

    @abc.abstractmethod
    def update(self):
        """Update the weights and return the next expected reward."""
        pass    # pylint: disable=W0107:unnecessary-pass

    @staticmethod
    def learning_process(func):
        """Decorator for a learning process that should repeat until cancelled."""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.thread:
                self.stop()
            self.thread = RepeatingThread(target=func, args=(self, *args), kwargs=kwargs)
            self.thread.start()
        return wrapper

    @learning_process
    def learn(self, iterations, interval: float = 0.1):
        """Learn a policy by interacting with the enviroment."""
        # establish basic elements
        returns = self.adapter.get_returns()

        self.next_action, self.exp_reward = self.selector.choose()
        for _ in range(iterations):
            self.action = self.next_action
            self.logger.debug("Chosen action: %s", self.action)
            prev_returns = returns
            self.selector.take_action(self.action)
            time.sleep(interval)
            returns = self.adapter.get_returns()
            self.last_reward = returns - prev_returns
            self.last_exp_reward = self.exp_reward
            self.next_action, self.exp_reward = self.selector.choose()
            self.update()

        # print results
        self.selector.log_values()
