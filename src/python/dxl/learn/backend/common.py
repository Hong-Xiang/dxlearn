class Backend:
    _default_backend = None

    def set_as_default(self):
        self._default_backend = self

    @classmethod
    def reset_default(cls):
        cls._default_backend = None

    def sandbox(self):
        """
        Context manager which provide a "sandbox" environment which can be played
        without afftect external environment.
        Useful for unittest.
        """
        raise NotImplementedError

    def unbox(self):
        raise NotImplementedError

    def maybe_unbox(self, t):
        raise NotImplementedError

    @classmethod
    def TestCase(cls):
        raise NotImplementedError

    def __getattr__(self, name):
        return getattr(self.unbox(), name)
