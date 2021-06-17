import pygame
from PIL import Image
import time
from math import sin, cos, atan2
import numpy
import random
import os
from collections import defaultdict
from bots import RandomBot
from constants import *
from helper import *


class Simulator:
    def __init__(self):
        self.draw_data = {}
        self.agent_control = False
        self.show_game = True

        # game shit
        self.running = True
        pygame.init()
        self.screen = pygame.display.set_mode([width, height])

        # first team is blue
        self.teams = tuple(tuple(RandomBot(team, ptype) for ptype in range(3)) for team in range(2))
        self.bullets = []  # bullet = team, type, pos, dir

        # sprites
        self.char_img = (
            (
                pygame.image.load("C:/Users/adama/OneDrive/Desktop/dyna_go/octorok_blue.png"),
                pygame.image.load("C:/Users/adama/OneDrive/Desktop/dyna_go/octorok_red.png")
            ),
            (
                pygame.image.load("C:/Users/adama/OneDrive/Desktop/dyna_go/black_mage_blue.png"),
                pygame.image.load("C:/Users/adama/OneDrive/Desktop/dyna_go/black_mage_red.png")
            ),
            (
                pygame.image.load("C:/Users/adama/OneDrive/Desktop/dyna_go/fairy_blue.png"),
                pygame.image.load("C:/Users/adama/OneDrive/Desktop/dyna_go/fairy_red.png")
            )
        )
        self.bullet_img = (
            pygame.image.load("C:/Users/adama/OneDrive/Desktop/dyna_go/octorok_bullet.png"),
            pygame.image.load("C:/Users/adama/OneDrive/Desktop/dyna_go/black_mage_bullet.png"),
            pygame.image.load("C:/Users/adama/OneDrive/Desktop/dyna_go/fairy_heal.png"),
        )

        # board shit
        self.move_delta = ((-1, 0), (1, 0), (0, -1), (0, 1))
        board_image = Image.open("C:/Users/adama/OneDrive/Desktop/dyna_go/dyna_go.png")
        self.board_image = pygame.image.load("C:/Users/adama/OneDrive/Desktop/dyna_go/dyna_go.png")
        self.board_image = pygame.transform.scale(self.board_image, (width, height))
        self.board = tuple(tuple(x) for x in numpy.array(board_image))
        board_image.close()

        def ok(y_, x_):
            return 0 <= y_ < board_height and 0 <= x_ < board_width

        delta = ((-1, 0), (1, 0), (0, -1), (0, 1))

        def all_neighbors(y_, x_):
            return ((y_ + dy, x_ + dx) for dy, dx in delta)

        def neighbors(y_, x_):
            return ((y_ + dy, x_ + dx) for dy, dx in delta if ok(y_ + dy, x_ + dx))

        def on_border(y_, x_):
            return [i for i, (yn_, xn_) in enumerate(all_neighbors(y_, x_)) if ok(yn_, xn_) and self.board[yn_][xn_]]

        self.borders = []
        seen = set()
        for y in range(board_height):
            for x in range(board_width):
                if self.board[y][x] or (y, x) in seen or not on_border(y, x):
                    continue
                seen.add((y, x))
                cmp = []
                for d in on_border(y, x):
                    cmp.append((d, y, x))
                q = [(y, x)]
                while q:
                    yc, xc = q.pop()
                    for yn, xn in neighbors(yc, xc):
                        if self.board[yn][xn] or (yn, xn) in seen or not on_border(yn, xn):
                            continue
                        seen.add((yn, xn))
                        q.append((yn, xn))
                        for d in on_border(yn, xn):
                            cmp.append((d, yn, xn))
                cmp.sort()

                def proc(first_, last_):
                    if first_[0] == 0:
                        self.borders.append((first_[1], first_[2], last_[1], last_[2] + 1))
                    elif first_[0] == 1:
                        self.borders.append((first_[1] + 1, first_[2], last_[1] + 1, last_[2] + 1))
                    elif first_[0] == 2:
                        self.borders.append((first_[1], first_[2], last_[1] + 1, last_[2]))
                    else:
                        self.borders.append((first_[1], first_[2] + 1, last_[1] + 1, last_[2] + 1))

                first, last = cmp[0], cmp[0]
                for d, yn, xn in cmp:
                    if last[0] != -1 and (last[0] != d or (last[1] != yn and last[2] != xn)):
                        proc(first, last)
                        first = [d, yn, xn]
                    last = [d, yn, xn]
                proc(first, last)

        # test if lines loaded correctly
        # self.screen.fill((255, 255, 255))
        # for border in self.borders:
        #     border = tuple(val * cell_size for val in border)
        #     pygame.draw.line(self.screen, (0, 0, 0), border[3:1:-1], border[1::-1], 1)
        # pygame.display.flip()
        # time.sleep(100)

    @staticmethod
    def bullet_hit(agent, bullet_type, bullet_pos, bullet_dir):
        if dist(add_tuple(bullet_pos, bullet_dir), agent) < dist(bullet_pos, agent):
            p0 = agent
            p1 = bullet_pos
            p2 = [bullet_pos[0] + bullet_dir[0], bullet_pos[1] + bullet_dir[1]]
            # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points
            min_dist = abs(p1[0] * (p2[1] - p0[1]) + p2[0] * (p0[1] - p1[1]) + p0[0] * (p1[1] - p2[1]))
            if min_dist < (adjusted_agent_radius + 0.2) * bullet_speed[bullet_type]:
                return True
        return False

    def bullet_state(self, agent):
        for bullet in self.bullets:
            if bullet[0] != agent.team and self.bullet_hit((agent.y, agent.x), *bullet[1:]):
                return 1 + round_ang((agent.y, agent.x), bullet[2], 4)
        return 0

    def episode(self):
        for team in self.teams:
            for agent in team:
                agent.reset()

        print(self.teams)
        agents = tuple(agent for team in self.teams for agent in team)
        self.bullets = []
        capture_frames = [0, 0]

        for frame in range(game_length_frames):
            states = []
            friends, enemies = [], []
            for idx, agent in enumerate(agents):
                if agent.reload or not agent.health:
                    states.append((4,))
                    continue

                visible = [[], []]
                for oidx, oagent in enumerate(agents):
                    if agent == oagent:
                        continue
                    l1 = (agent.y + 0.5, agent.x + 0.5, oagent.y + 0.5, oagent.x + 0.5)
                    # for l2 in self.borders:        #  good debug if visibility check is wrong
                    #     if intersect(l1, l2) ^ intersect(l2, l1):
                    #         print(l1, l2, math.int)
                    if not any(intersect(l1, l2) for l2 in self.borders):
                        visible[oidx // 3].append((dist((agent.y, agent.x), (oagent.y, oagent.x)), oagent))
                if idx >= 3:
                    visible = [visible[1], visible[0]]
                if not frame:
                    print(agent, visible)
                state_type = bool(visible[0]) + 2 * bool(visible[1])
                friend = max(visible[0], key=lambda x: x[0])[1] if visible[0] else None
                enemy = max(visible[1], key=lambda x: x[0])[1] if visible[1] else None
                friends.append(friend)
                enemies.append(enemy)

                self_info = (
                    agent.y, agent.x,
                    2 if agent.cd else (0 if not agent.ammo else 1),
                    int(agent.health >= health_thresh),
                    self.bullet_state(agent)
                )

                states.append((state_type, (*self_info, *friend_state(agent, friend), *enemy_state(agent, enemy))))

            pygame.event.get()

            # actions
            for state, agent, friend, enemy in zip(states, agents, friends, enemies):
                if not agent.health:
                    continue
                action = agent.action(state)
                if 0 <= action <= 3:
                    if not agent.reload:
                        target = agent.y + self.move_delta[action][0], agent.x + self.move_delta[action][1]
                        if self.board[target[0]][target[1]] != 0:
                            agent.y, agent.x = target
                elif agent.ammo and not agent.cd:
                    if agent.ptype == 2:  # healer
                        if friend:
                            friend.health = min(100, friend.health + bullet_damage[2])
                    elif enemy:
                        angle = atan2(enemy.y - agent.y, enemy.x - agent.x)
                        bullet = [agent.team, agent.ptype, [agent.y + 0.5, agent.x + 0.5], (sin(angle), cos(angle))]
                        self.bullets.append(bullet)
                elif not agent.ammo:
                    agent.reload = reload[agent.ptype]

            # environment response
            rewards = [0.] * 6

            # bullets
            rem_bullets = []
            for bullet in self.bullets:
                bullet[2][0] += bullet[3][0] * bullet_speed[bullet[1]]
                bullet[2][1] += bullet[3][1] * bullet_speed[bullet[1]]
                hit = False

                for agent in agents:
                    if agent.y == int(bullet[2][0]) and agent.x == int(bullet[2][1]) and agent.team != bullet[0]:
                        hit = True
                        agent.health = max(0, agent.health - bullet_damage[bullet[1]])

                # print(bullet[2])
                if not hit and self.board[int(bullet[2][0])][int(bullet[2][1])] != 0:
                    rem_bullets.append(bullet)

            self.bullets = rem_bullets

            # capture
            for i in range(2):
                captured = any(
                    capture_y[i][0] <= agent.y <= capture_y[i][1] and capture_x[i][0] <= agent.x <= capture_x[i][1]
                    for agent in self.teams[1]
                )
                capture_frames[i] = capture_frames[i] + 1 if captured else 0
                if capture_frames[i] == capture_thresh:
                    return 1

            # elimination
            for idx, team in enumerate(self.teams):
                if all(agent.health == 0 for agent in team):
                    return 1 - idx
            if self.show_game:
                if frame:
                    self.draw()
                self.draw_data = self.get_draw_data()
        return -1

    def get_draw_data(self):
        data = {}
        for team in self.teams:
            for agent in team:
                data[agent] = (agent.x, agent.y)
        return data

    def draw(self):
        for t in range(iframes):
            self.screen.blit(self.board_image, (0, 0))
            for team in self.teams:
                for agent in team:
                    self.screen.blit(self.char_img[agent.ptype][agent.team], (
                        ((t / iframes) * agent.x + (1 - t / iframes) * self.draw_data[agent][0]) * cell_size,
                        ((t / iframes) * agent.y + (1 - t / iframes) * self.draw_data[agent][1]) * cell_size,
                    ))
            for bullet in self.bullets:
                if self.board[int(bullet[2][0] + (t / iframes) * bullet[3][0] * bullet_speed[bullet[0]])][
                        int(bullet[2][1] + (t / iframes) * bullet[3][1] * bullet_speed[bullet[1]])] == 0:
                    continue
                # print(bullet)
                # print((bullet[2][0] + (t / 10) * bullet[3][0] * bullet_speed[bullet[0]]) * cell_size,
                #       (bullet[2][1] + (t / 10) * bullet[3][1] * bullet_speed[bullet[1]]) * cell_size, "gaming")
                self.screen.blit(self.bullet_img[bullet[1]], (
                    (bullet[2][1] + (t / iframes) * bullet[3][1] * bullet_speed[bullet[1]]) * cell_size,
                    (bullet[2][0] + (t / iframes) * bullet[3][0] * bullet_speed[bullet[0]]) * cell_size
                ))

            pygame.display.flip()
            time.sleep(1 / (iframes * fps))


simul = Simulator()

print(simul.episode())

# for i in range(1000):
# if i == 0:
#     print("swapped")
#     simul.teams[0] = ManualBot()
# simul.teams[0].epsilon *= 0.9
# simul.teams[1].epsilon *= 0.9
#
# wins = 0
# ties = 0
# games = 0
#
# f, s = 0., 0.
#
# for _ in range(2000):
#     start = time.time()
#     e1, e2, w = simul.episode()
#     f += time.time() - start
#     games += 1
#     wins += w == 1
#     ties += w == 2
#     simul.teams[0].update(*e1)
#     start = time.time()
#     simul.teams[1].update(*e2)
#     s += time.time() - start
# # print(f, s)
#
# print("win rate: ", wins / games)
# print("tie rate: ", ties / games)
# print()
#
# simul.show_game = True
# simul.episode()
# simul.show_game = False

pygame.quit()
