"""A module for the state of the game."""


import numpy as np

from paperclips_ai.adapter import WebAdapter, WebElement, EmptyWebElement

class State:
    """A class representing the state of the game."""
    def __init__(self, adapter: WebAdapter, features: set[str]):
        """Initializes the state.

        Args:
            adapter: The adapter for the game.
            features: The ids of all features to track.
        """
        self.adapter = adapter
        self.size = len(features) + 1
        self.vector = np.zeros(self.size)
        self.last_reward: int = 0

        features.add('clips')
        self.features = (
            [EmptyWebElement('bias')]
            + [self.adapter.get_elem_by_id(feature)
               for feature in features]
        )
        self.clips_el: WebElement = [feat for feat in self.features if feat.name == 'clips'][0]

        self.update()

    def update(self):
        """Read data from webpage to determine current state."""
        values = [np.log10(feature.value) if feature.value > 0 else 0
                  for feature in self.features]
        self.vector = np.array(values)

    @property
    def returns(self) -> float:
        """The total returns of the game."""
        return self.clips_el.value
