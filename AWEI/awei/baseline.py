from __future__ import annotations

import numpy as np
from awei.abstract_policy import AbstractPolicy
from awei.custom_types import Action, Proportion, State

# prototype = [
#     (0, 0.3),
#     (0.4, 0.6),
#     (0.2, 0.1)
# ]


def make_schedule(prototype: list[tuple[Action, Proportion]], n_actions: int) -> list:
    prototype = np.array(prototype)

    assert np.isclose(np.sum(prototype[:, 1]), 1), "Schedule proportions do not sum to 1"

    actions = prototype[:, 0]
    proportions = prototype[:, 1]
    repeats = (proportions * n_actions).astype(int)

    # Fill last if length does not quite match
    diff = n_actions - np.sum(repeats)
    if 0 < diff <= 2:
        repeats[-1] += diff

    schedule = np.repeat(actions, repeats)

    assert len(schedule) == n_actions, f"Length of schedule ({len(schedule)}) is longer than number of actions ({n_actions})."

    return list(schedule)


class Schedule(AbstractPolicy):
    def __init__(self, schedule: list[float]) -> None:
        super().__init__()
        self.schedule = schedule
        self.schedule_iterator = iter(schedule)

    def get_next_action(self):
        return next(self.schedule_iterator)

    def act(self, state: State) -> Action:
        return self.get_next_action()

    def reset(self):
        self.schedule_iterator = iter(self.schedule)
