import random
import numpy
from constants import *
import torch


class Player:
    def __init__(self, team, ptype, k):
        self.team, self.ptype = team, ptype
        self.k = k
        self.states = torch.empty((game_length_frames, k), dtype=torch.int, device=dev)
        self.actions = torch.empty((game_length_frames, k), dtype=torch.int, device=dev)
        self.rewards = torch.zeros((game_length_frames, k), dtype=torch.float16, device=dev)
        self.epsilon = 0.2
        self.alpha = 0.2

    def reset(self):
        self.rewards = torch.zeros((game_length_frames, self.k), dtype=torch.float16, device=dev)

    def update(self):
        pass


class RandomBot(Player):
    def __init__(self, team, ptype, k):
        super().__init__(team, ptype, k)

    def action(self, state, frame):
        return torch.randint(0, 4, (self.k,), device=dev)


class MCBot(Player):
    def __init__(self, team, ptype, k):
        super().__init__(team, ptype, k)
        self.q = 0.1 * torch.rand((total_states, 5), device=dev)
        self.q[-5:] = 0.
        self.epsilon = 0.2
        self.gamma = 0.9
        self.alpha = 0.2
        self.greedy = False
        self.reset()

    def action(self, state, frame):
        self.states[frame] = state
        action = torch.where(
            torch.rand((self.k,), device=dev) < self.epsilon,
            torch.randint(5, (self.k,), device=dev),
            torch.argmax(self.q[state], dim=1)
        )
        self.actions[frame] = action
        return action

    def update(self):
        g = torch.zeros((self.k,), dtype=torch.float, device=dev)
        for frame in range(game_length_frames - 1, -1, -1):
            g = self.gamma * g + self.rewards[frame]
            s, a = self.states[frame], self.actions[frame]
            self.q[s, a] += self.alpha * (g - self.q[s, a])


class TDBot(Player):
    def __init__(self, team, ptype, k):
        super().__init__(team, ptype, k)
        self.q = 0.1 * torch.rand((total_states, 5), device=dev)
        self.q[-5:] = 0.
        self.epsilon = 0.2
        self.gamma = 0.8
        self.alpha = 0.4
        self.greedy = False
        self.reset()

    def action(self, state, frame):
        if frame:  # TODO: figure out if need to do this update after last game state too (shouldn't matter unless tie penalty ?)
            s = self.states[frame - 1]
            a = self.actions[frame - 1]
            r = self.rewards[frame - 1]
            sp = state
            self.q[s, a] += self.alpha * (r + self.gamma * self.q[sp].max(dim=1)[0] - self.q[s, a])
        self.states[frame] = state
        action = torch.where(
            torch.rand((self.k,), device=dev) < self.epsilon,
            torch.randint(5, (self.k,), device=dev),
            torch.argmax(self.q[state], dim=1)
        )
        self.actions[frame] = action
        return action

    def update(self):
        pass
