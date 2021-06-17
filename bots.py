import random
from constants import *


class Player:
    def __init__(self, team, ptype):
        self.team, self.ptype = team, ptype
        self.x, self.y = 0, 0
        self.ammo, self.reload, self.cd = 0, 0, 0
        self.health = 100
        self.states, self.actions, self.rewards = [], [], []

    def reset(self):
        self.x, self.y = random.randint(*spawn_x[self.team]), random.randint(*spawn_y[self.team])
        self.ammo, self.reload, self.cd = max_ammo[self.ptype], 0, 0
        self.health = 100
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

    def __str__(self):
        return [
            ["Blue", "Red"][self.team],
            ["Sniper", "Gamer", "Healer"][self.ptype],
            "HP: " + str(self.health)
        ].__str__()

    def __repr__(self):
        return " ".join([
            ["Blue", "Red"][self.team],
            ["sniper,", "gamer,", "healer,"][self.ptype],
            "HP: " + str(self.health)
        ])


class RandomBot(Player):
    def __init__(self, team, ptype):
        super().__init__(team, ptype)

    def action(self, state):
        return random.randint(0, 4)
