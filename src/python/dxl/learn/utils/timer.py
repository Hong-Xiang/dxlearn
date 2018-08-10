import arrow


class Repeater:
    def __init__(self, interval, func):
        self.interval = interval
        self.func = func
        self._previous = None

    def is_run_func(self):
        pass

    def update_previous(self):
        pass

    def run(self):
        if self.is_run_func():
            self.update_previous()
            return self.func()


class TimeRepeater(Repeater):
    def is_run_func(self):
        return self._previous is None or (
            arrow.now() - self._previous).secounds > self.interval

    def update_previous(self):
        self._previous = arrow.now()


class GlobalStepRepeater(Repeater):
    def is_run_func(self):
        pass

    def update_previous(self):
        pass
