import lab as B
from .dist import AbstractDistribution

__all__ = ["ReciprocalInt"]


class ReciprocalInt(AbstractDistribution):

    def __init__(self, lower: B.Int, upper: B.Int, smoothing: B.Float = 1.0):
        self.lower = lower
        self.upper = upper
        self.smoothing = smoothing
        self.values = B.range(self.lower, self.upper + 1, 1)
        self.probs = 1.0 / B.maximum(self.values.astype(float) ** self.smoothing, 1)
        self.probs = self.probs / self.probs.sum()

    def sample(self, state, dtype, *shape):
        B.set_global_random_state(state)
        sample = B.choice(self.values, *shape, p=self.probs)
        print(sample)
        # state = B.set_global_random_state()
        # return B.randint(state, dtype, lower=self.lower, upper=self.upper + 1, *shape)