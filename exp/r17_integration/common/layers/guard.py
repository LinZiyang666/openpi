"""Only-no-progress: R8/K1 span arithmetic, without gripper/task assumptions."""


class NoProgressGuard:
    def __init__(self, *, n=3, eps=.5):
        if n < 2 or eps < 0:
            raise ValueError("invalid no-progress thresholds")
        self.n, self.eps = n, eps
        self.reset()

    def reset(self):
        self.history = []
        self.span = 0

    def observe(self, step, progress, episode_length, *, prev_hit=None):
        self.history = [x for x in self.history if x[0] < step]
        if prev_hit is False:
            self.history = []
        self.history.append((step, float(progress), max(int(episode_length) - 1, 1)))
        span = 0
        for prev, now in zip(self.history[:-1], self.history[1:]):
            span = span + now[0] - prev[0] if (now[1] - prev[1]) * now[2] <= self.eps else 0
        self.span = span
        return span >= self.n - 1

    def veto_tail(self):
        # A blind interval reaching the trigger span must LOOK before serving.
        return self.span >= self.n - 2
