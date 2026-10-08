"""Test-only equal metadata whose alias identity depends on unpickle order."""


class SyntheticPickleMemo:
    cached = None

    def __init__(self):
        self.token = [1, 2, 3]
        self.alias = self.token

    def __setstate__(self, state):
        self.__dict__.update(state)
        if type(self).cached is None:
            type(self).cached = self.token
        self.alias = type(self).cached
