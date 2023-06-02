from __future__ import annotations

from abc import abstractmethod

from awei.custom_types import Action, State


class AbstractPolicy(object):
    @abstractmethod
    def act(self, state: State) -> Action:
        raise NotImplementedError
