from paperclips_ai import WebAdapter
import numpy as np

class State:
    """A class representing the state of the game."""
    def __init__(self, adapter: WebAdapter, features: list[str]):
        self.adapter = adapter
        self.size = len(features) + 1
        self.vector = np.zeros(self.size)

        self.features = [
            self.adapter.get_elem_by_id(feature)
            for feature in features]
        self.features.insert(0, None)

        self._update()

    def _update(self):
        """Read data from webpage to determine current state."""
        # get values
        values = [1 if feature is None else feature.value for feature in self.features]
        self.vector = np.array(values)

    @property
    def returns(self):
        """The total returns of the game."""
        return self.adapter.get_returns()

    def update(self):
        """Update the state and get the last reward."""
        old_return = self.returns
        self._update()
        return self.returns - old_return
