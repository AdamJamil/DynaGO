import time
from math import sin, cos, atan2

import pygame
from PIL import Image

from bots import *
from helper import *


class Simulator:
    def __init__(self):
        self.show_game = True
        self.draw_data = {}

        # game shit
        self.running = True
        pygame.init()
        self.screen = pygame.display.set_mode([width, height])

        # first team is blue
        self.teams = tuple((
            tuple(TDBot(0, ptype) for ptype in team_types),
            tuple(RandomBot(1, ptype) for ptype in team_types)
        ))
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
        self.bullet_sz = [(img.get_height(), img.get_width()) for img in self.bullet_img]

        # board shit
        self.move_delta = ((-1, 0), (1, 0), (0, -1), (0, 1))
        board_image = Image.open("C:/Users/adama/OneDrive/Desktop/dyna_go/dyna_small.png")
        self.board_image = pygame.image.load("C:/Users/adama/OneDrive/Desktop/dyna_go/dyna_small.png")
        self.board_image = pygame.transform.scale(self.board_image, (width, height))
        if len(numpy.array(board_image).shape) == 3:
            self.board = tuple(tuple(int(val[0] != 0) for val in x) for x in numpy.array(board_image))
        else:
            self.board = tuple(tuple(x) for x in numpy.array(board_image))
        board_image.close()

        delta = ((-1, 0), (1, 0), (0, -1), (0, 1))

        def all_neigh(y_, x_):
            return ((y_ + dy, x_ + dx) for dy, dx in delta)

        def neighbors(y_, x_):
            return ((y_ + dy, x_ + dx) for dy, dx in delta if self.ok(y_ + dy, x_ + dx))

        def on_border(y_, x_):
            return [i for i, (yn_, xn_) in enumerate(all_neigh(y_, x_)) if self.ok(yn_, xn_) and self.board[yn_][xn_]]

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

        self.cell_idx = [[-1] * board_width for _ in range(board_height)]
        curr = 0
        for y in range(board_height):
            for x in range(board_width):
                if self.board[y][x]:
                    self.cell_idx[y][x] = curr
                    curr += 1
        print(curr, "gaming")

        # test if lines loaded correctly
        # self.screen.fill((255, 255, 255))
        # for border in self.borders:
        #     border = tuple(val * cell_size for val in border)
        #     pygame.draw.line(self.screen, (0, 0, 0), border[3:1:-1], border[1::-1], 1)
        # pygame.display.flip()
        # time.sleep(100)

    @staticmethod
    def ok(y_, x_):
        return 0 <= y_ < board_height and 0 <= x_ < board_width

    def is_visible(self, p1, p2):
        l1 = (p1[0], p1[1], p2[0], p2[1])
        is_vis = True
        for l2 in self.borders:
            if intersect(l1, l2):
                is_vis = False
        return is_vis

    def bullet_hit(self, agent, bullet_type, bullet_pos, bullet_dir):
        if not self.is_visible(agent, bullet_pos):
            return False
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
            if bullet[0] != agent.team and self.bullet_hit((agent.y + 0.5, agent.x + 0.5), bullet[1], bullet[2], bullet[3]):
                return 1 + round_ang((agent.y + 0.5, agent.x + 0.5), bullet[2], 4)
        return 0

    def episode(self):
        agents = []
        for team in self.teams:
            for agent in team:
                agent.reset()
                agents.append(agent)


        self.bullets = []
        # capture_frames = [0, 0]

        for frame in range(game_length_frames):
            states = []
            friends, enemies = [], []

            for bullet in self.bullets:
                bullet[2][0] += bullet[3][0]
                bullet[2][1] += bullet[3][1]

            for agent in agents:
                if agent.reload or not agent.health:
                    states.append((4, (1 if not agent.health else 0)))
                    friends.append(None)
                    enemies.append(None)
                    continue

                visible = [[], []]
                for oagent in agents:
                    if agent == oagent or not oagent.health:
                        continue
                    # for l2 in self.borders:        #  good debug if visibility check is wrong
                    #     if intersect(l1, l2) ^ intersect(l2, l1):
                    #         print(l1, l2, math.int)
                    if self.is_visible((agent.y + 0.5, agent.x + 0.5), (oagent.y + 0.5, oagent.x + 0.5)):
                        visible[oagent.team].append((dist((agent.y, agent.x), (oagent.y, oagent.x)), oagent))
                if agent.team == 1:
                    visible = [visible[1], visible[0]]

                assert(sum(len(x) for x in visible) == sum(agent.health != 0 for agent in agents) - 1)

                visible = [sorted(visible[0], key=lambda x: x[0]), sorted(visible[1], key=lambda x: x[0])]  # TODO: remove for optimziation
                if self.show_game and agent.team == 0 and agent.ptype == 1:
                    self.lines = []
                    self.olines = []
                    for vfriend in visible[0]:
                        self.lines.append(((agent.x + 0.5, agent.y + 0.5), (vfriend[1].x + 0.5, vfriend[1].y + 0.5)))
                    for venemy in visible[1]:
                        self.olines.append(((agent.x + 0.5, agent.y + 0.5), (venemy[1].x + 0.5, venemy[1].y + 0.5)))

                state_type = bool(visible[0]) + 2 * bool(visible[1])
                friend = min(visible[0], key=lambda x: x[0])[1] if visible[0] else None
                enemy = min(visible[1], key=lambda x: x[0])[1] if visible[1] else None
                friends.append(friend)
                enemies.append(enemy)

                self_info = (((self.cell_idx[agent.y][agent.x] * 3 +
                               (2 if agent.cd else (0 if not agent.ammo else 1))) * 2 +
                              int(agent.health >= health_thresh)) * 5 +
                             self.bullet_state(agent), )

                states.append((state_type, (*self_info, *friend_state(agent, friend), *enemy_state(agent, enemy))))

            pygame.event.get()

            rewards = [0.] * 6

            # actions
            to_shoot = []
            for state, agent, friend, enemy in zip(states, agents, friends, enemies):
                action = agent.action(state)
                if not agent.health:
                    continue

                if not agent.reload:
                    if 0 <= action <= 3:
                        target = agent.y + self.move_delta[action][0], agent.x + self.move_delta[action][1]
                        if self.board[target[0]][target[1]] != 0:
                            agent.y, agent.x = target
                    elif agent.ammo and not agent.cd:
                        if agent.ptype == 2:  # healer
                            if friend and dist((friend.y, friend.x), (agent.y, agent.x)) < distance_thresh:
                                heal = min(bullet_damage[2], 100 - friend.health)
                                rewards[3 * friend.team + friend.ptype] += heal * damage_reward_ratio
                                rewards[3 * agent.team + agent.ptype] += heal * damage_reward_ratio
                                friend.health += heal
                                agent.cd = cd[2]
                        elif enemy and agent.ptype == 1:
                            to_shoot.append((enemy, agent.ptype, (agent.y, agent.x)))
                            agent.ammo += -1
                            agent.cd = cd[agent.ptype]
                    elif not agent.ammo:
                        agent.reload = reload[agent.ptype]
                if agent.reload == 1:
                    agent.ammo = max_ammo[agent.ptype]
                agent.reload = max(0, agent.reload - 1)
                agent.cd = max(0, agent.cd - 1)

            for enemy, ptype, (y, x) in to_shoot:
                angle = atan2(enemy.y - y, enemy.x - x)
                angle += (random.random() * 2 * bullet_spread[agent.ptype]) - bullet_spread[agent.ptype]
                bullet = [1 - enemy.team, ptype, [y + 0.5, x + 0.5],
                                (sin(angle) * bullet_speed[ptype], cos(angle) * bullet_speed[ptype])]
                self.bullets.append(bullet)

            # environment response

            # bullets
            rem_bullets = []
            for bullet in self.bullets:
                hit = False

                for agent in agents:
                    if not agent.health or agent.team == bullet[0]:
                        continue
                    if dist((agent.y + 0.5, agent.x + 0.5), bullet[2]) <= adjusted_agent_radius:
                        hit = True
                        dmg = min(bullet_damage[bullet[1]], agent.health)
                        rewards[3 * bullet[0] + bullet[1]] += dmg * damage_reward_ratio
                        rewards[3 * agent.team + agent.ptype] -= dmg * damage_reward_ratio
                        agent.health -= dmg
                        if not agent.health:
                            rewards[3 * bullet[0] + bullet[1]] += kill_reward
                            rewards[3 * agent.team + agent.ptype] -= kill_reward

                if not hit and self.ok(int(bullet[2][0]), int(bullet[2][1])):
                    if self.board[int(bullet[2][0])][int(bullet[2][1])] != 0:
                        rem_bullets.append(bullet)

            self.bullets = rem_bullets

            done = 0

            # # capture
            # for i in range(2):
            #     captured = False
            #     for agent in self.teams[1]:
            #         if agent.health:
            #             if capture_y[i][0] <= agent.y <= capture_y[i][1] and capture_x[i][0] <= agent.x <= capture_x[i][1]:
            #                 captured = True
            #
            #     capture_frames[i] = capture_frames[i] + 1 if captured else 0
            #     if capture_frames[i] == capture_thresh:
            #         print("poggers")
            #         done = 2
            #         for j in range(3 * (done - 1), 3 * done):
            #             rewards[j] += win_reward
            #             rewards[(j + 3) % 6] -= win_reward

            # elimination
            for idx, team in enumerate(self.teams):
                all_dead = True
                for agent in team:
                    all_dead &= agent.health == 0
                if all_dead:
                    done = 2 - idx
                    for i in range(team_size * (done - 1), team_size * done):
                        rewards[i] += win_reward
                        rewards[(i + team_size) % (team_size * 2)] -= win_reward

            if self.show_game:
                if frame:
                    self.draw()
                self.set_draw_data()

            for r, agent in zip(rewards, agents):
                agent.rewards.append(r - tie_penalty * (frame == game_length_frames - 1 and not done))

            if done:
                return done
        return -1

    def set_draw_data(self):
        for team in self.teams:
            for agent in team:
                self.draw_data[agent] = (agent.x, agent.y)

    def draw(self):
        for t in range(iframes):
            self.screen.blit(self.board_image, (0, 0))
            if not t:
                pygame.draw.rect(self.screen, (255, 0, 0), ((0, 0), (20, 20)))
            for team in self.teams:
                for agent in team:
                    if not agent.health:
                        continue
                    ax = ((t / iframes) * agent.x + (1 - t / iframes) * self.draw_data[agent][0]) * cell_size
                    ay = ((t / iframes) * agent.y + (1 - t / iframes) * self.draw_data[agent][1]) * cell_size
                    pygame.draw.rect(self.screen, (255, 0, 0), ((ax, ay - 10), (24, 8)))
                    pygame.draw.rect(self.screen, (0, 255, 0), ((ax, ay - 10), (24 * agent.health / 100, 8)))
                    self.screen.blit(self.char_img[agent.ptype][agent.team], (ax, ay))
                    if agent.reload:
                        pygame.draw.rect(self.screen, (50, 50, 50),
                                         ((ax - 7, ay + (agent.reload / reload[agent.ptype]) * 24),
                                          (4, 24 - (agent.reload / reload[agent.ptype]) * 24)))
                    if agent.states and agent.states[-1][0] != 4 and agent.states[-1][1][0] % 5:
                        pygame.draw.rect(self.screen, (255, 0, 255), ((ax, ay + 30), (27, 5)))

            for bullet in self.bullets:
                by = bullet[2][0] + (t / iframes) * bullet[3][0] - bullet[3][0]
                bx = bullet[2][1] + (t / iframes) * bullet[3][1] - bullet[3][1]
                if self.board[int(by)][int(bx)] == 0:
                    continue
                self.screen.blit(self.bullet_img[bullet[1]], (
                    bx * cell_size - self.bullet_sz[bullet[1]][1] / 2,
                    by * cell_size - self.bullet_sz[bullet[1]][0] / 2,
                ))
            for agent in self.teams[0]:
                if agent.ptype == 1 and agent.health:
                    for idx, line in enumerate(self.lines):
                        pygame.draw.line(self.screen, (255 * (idx == 0), 50, 255), *((point[0] * cell_size, point[1] * cell_size) for point in line))

                    for idx, line in enumerate(self.olines):
                        pygame.draw.line(self.screen, (255, 50, 255 * (idx == 0)), *((point[0] * cell_size, point[1] * cell_size) for point in line))
                    break

            pygame.display.flip()
            time.sleep(1 / (iframes * fps))


simul = Simulator()
simul.show_game = False

# simul.show_game = True
# simul.episode()
# simul.show_game = False

for i_ in range(1000):
    for team_ in simul.teams:
        for agent_ in team_:
            agent_.epsilon *= 0.98
            agent_.alpha *= 0.98

    wins = 0
    ties = 0
    games_ = 100

    for game_ in range(games_):
        # if game % (games // 10) == 0:
        #     print(str(10 * game / (games // 10)) + "%")
        start = time.time()
        w = simul.episode()
        wins += w == 1
        ties += w == -1
        for team_ in simul.teams:
            for agent_ in team_:
                agent_.update()

    print("blue win rate: ", wins / games_)
    print("tie rate: ", ties / games_)
    print("red win rate: ", (games_ - wins - ties) / games_)
    print("eps=" + str(simul.teams[0][0].epsilon))
    print()

    simul.show_game = True
    simul.episode()
    simul.show_game = False

pygame.quit()
