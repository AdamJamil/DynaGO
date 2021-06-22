import random
import numpy
from constants import *


class Player:
    def __init__(self, team, ptype):
        self.team, self.ptype = team, ptype
        self.x, self.y = 0, 0
        self.ammo, self.reload, self.cd = 0, 0, 0
        self.health = 100
        self.states, self.actions, self.rewards = [], [], []
        self.epsilon = 0.2
        self.alpha = 0.2

    def reset(self):
        self.x, self.y = random.randint(*spawn_x[self.team]), random.randint(*spawn_y[self.team])
        self.ammo, self.reload, self.cd = max_ammo[self.ptype], 0, 0
        self.health = 100
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

    def update(self):
        pass

    def __str__(self):
        return " ".join([
            ["Blue", "Red"][self.team],
            ["sniper,", "gamer,", "healer,"][self.ptype],
            "HP: " + str(self.health),
            "Reload: " + str(self.reload)
        ])

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


class MCBot(Player):
    def __init__(self, team, ptype):
        super().__init__(team, ptype)

        self.q = [
            2 * numpy.random.random((self_info_size, 5)),
            2 * numpy.random.random((self_info_size, friend_info_size, 5)),
            2 * numpy.random.random((self_info_size, enemy_info_size, 5)),
            2 * numpy.random.random((self_info_size, friend_info_size, enemy_info_size, 5)),
            2 * numpy.random.random((2, 1))
        ]
        self.epsilon = 0.2
        self.gamma = 0.9
        self.alpha = 0.2
        self.greedy = False
        self.reset()

    def action(self, state):
        self.states.append(state)
        if state[0] == 4:
            self.actions.append(0)
        elif numpy.random.random(1)[0] < self.epsilon and not self.greedy:
            self.actions.append(random.randint(0, 4))  # inclusive
        else:
            self.actions.append(numpy.argmax(self.q[state[0]][state[1]]))
        return self.actions[-1]

    def update(self):
        g = 0
        g_ = []
        for s, a, r in list(zip(self.states, self.actions, self.rewards))[::-1]:
            g = self.gamma * g + r
            g_.append(g)
            self.q[s[0]][s[1]][a] += self.alpha * (g - self.q[s[0]][s[1]][a])


class TDBot(Player):
    def __init__(self, team, ptype):
        super().__init__(team, ptype)

        self.q = [
            2 * numpy.random.random((self_info_size, 5)),
            2 * numpy.random.random((self_info_size, friend_info_size, 5)),
            2 * numpy.random.random((self_info_size, enemy_info_size, 5)),
            2 * numpy.random.random((self_info_size, friend_info_size, enemy_info_size, 5)),
            2 * numpy.random.random((2, 1))
        ]
        self.epsilon = 0.2
        self.gamma = 0.8
        self.alpha = 0.2
        self.greedy = False
        self.reset()

    def action(self, state):
        if self.states:
            s = self.states[-1]
            self.q[s[0]][s[1]][self.actions[-1]] += self.alpha * (
                    self.rewards[-1] + self.gamma * numpy.max(self.q[state[0]][state[1]])
                    - self.q[s[0]][s[1]][self.actions[-1]]
            )
        self.states.append(state)
        if state[0] == 4:
            self.actions.append(0)
        elif numpy.random.random(1)[0] < self.epsilon and not self.greedy:
            self.actions.append(random.randint(0, 4))  # inclusive
        else:
            self.actions.append(numpy.argmax(self.q[state[0]][state[1]]))
        return self.actions[-1]

    def update(self):
        pass