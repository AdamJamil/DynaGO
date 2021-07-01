import itertools
import time
from math import sin, cos, atan2

import pygame
from PIL import Image

from bots import *
from helper import *
from constants import *

import torch


class Simulator:
    def __init__(self, k):
        self.k = k
        self.show_game = True
        self.draw_data = {}

        # game shit
        self.running = True
        pygame.init()
        self.screen = pygame.display.set_mode([width, height])

        # first team is blue
        self.teams = tuple((
            tuple(TDBot(0, ptype, batch_size) for ptype in team_types),
            tuple(RandomBot(1, ptype, batch_size) for ptype in team_types)
        ))
        self.bullets = torch.tensor([])

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
        self.move_delta = torch.tensor(((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)), device=dev)
        board_image = Image.open("C:/Users/adama/OneDrive/Desktop/dyna_go/dyna_small.png")
        self.board_image = pygame.image.load("C:/Users/adama/OneDrive/Desktop/dyna_go/dyna_small.png")
        self.board_image = pygame.transform.scale(self.board_image, (width, height))
        if len(numpy.array(board_image).shape) == 3:
            self.board = tuple(tuple(int(val[0] != 0) for val in x) for x in numpy.array(board_image))
        else:
            self.board = tuple(tuple(x) for x in numpy.array(board_image))
        board_image.close()

        self.board = torch.tensor(self.board, dtype=torch.int, device=dev)

        delta = ((-1, 0), (1, 0), (0, -1), (0, 1))

        def all_neigh(y_, x_):
            return ((y_ + dy, x_ + dx) for dy, dx in delta)

        def neighbors(y_, x_):
            return ((y_ + dy, x_ + dx) for dy, dx in delta if self.ok(y_ + dy, x_ + dx))

        def on_border(y_, x_):
            return [o for o, (yn_, xn_) in enumerate(all_neigh(y_, x_)) if self.ok(yn_, xn_) and self.board[yn_][xn_]]

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

        self.cell_idx = torch.zeros((board_height, board_width), dtype=torch.long, device=dev)
        self.open_cell = []
        curr = 0
        for y in range(board_height):
            for x in range(board_width):
                if self.board[y][x]:
                    self.cell_idx[y][x] = curr
                    self.open_cell.append((y, x))
                    curr += 1
        print(f"Board has {curr} open cells.")
        self.is_visible = torch.empty((curr, curr), dtype=torch.bool, device=dev)
        self.dist = torch.empty((curr, curr), dtype=torch.half, device=dev)
        self.ang8 = torch.empty((curr, curr), dtype=torch.int8, device=dev)
        self.ang = torch.empty((curr, curr), dtype=torch.half, device=dev)
        for i in range(curr):
            for j in range(curr):
                self.is_visible[i][j] = self.is_visible_(
                    (self.open_cell[i][0] + 0.5, self.open_cell[i][1] + 0.5),
                    (self.open_cell[j][0] + 0.5, self.open_cell[j][1] + 0.5)
                )
                dy = self.open_cell[j][0] - self.open_cell[i][0]
                dx = self.open_cell[j][1] - self.open_cell[i][1]
                self.dist[i][j] = ((dy ** 2) + (dx ** 2)) ** 0.5
                self.ang[i][j] = math.atan2(dy, dx)
                self.ang8[i][j] = int(3 + 4 * (self.ang[i][j] / math.pi))

        # test if is_visible is correct
        # self.screen.fill((255, 255, 255))
        # for i in range(board_height):
        #     for j in range(board_width):
        #         pygame.draw.circle(self.screen, (0, 0, 255), (cell_size * (j + 0.5), cell_size * (i + 0.5)), 4)
        #         if self.board[i][j] == 0:
        #             print("h")
        #             pygame.draw.rect(self.screen, (0, 0, 0), ((cell_size * (j + 0.0), cell_size * (i + 0.0)), (cell_size, cell_size)))
        # for i in range(curr):
        #     for j in range(curr):
        #         if self.is_visible[i][j]:
        #             pygame.draw.line(self.screen, (255, 0, 0), (cell_size * (self.open_cell[i][1] + 0.5), (self.open_cell[i][0] + 0.5) * cell_size),
        #                              (cell_size * (self.open_cell[j][1] + 0.5), cell_size * (self.open_cell[j][0] + 0.5)))`

        # test if lines loaded correctly
        # self.screen.fill((255, 255, 255))
        # for border in self.borders:
        #     border = tuple(val * cell_size for val in border)
        #     pygame.draw.line(self.screen, (0, 0, 0), border[3:1:-1], border[1::-1], 1)
        # pygame.display.flip()
        # time.sleep(100)

    def is_visible_(self, p1, p2):
        l1 = (p1[0], p1[1], p2[0], p2[1])
        is_vis = True
        for l2 in self.borders:
            if intersect(l1, l2):
                is_vis = False
        return is_vis

    @staticmethod
    def ok(y_, x_):
        return 0 <= y_ < board_height and 0 <= x_ < board_width

    def episode(self):
        k = self.k
        agents = [agent for team in self.teams for agent in team]
        for agent in agents:
            agent.reset()

        pos = torch.empty((k, 2 * team_size, 2), dtype=torch.int64, device=dev)
        ammo = torch.empty((k, 2 * team_size), dtype=torch.int8, device=dev)
        reload = torch.zeros((k, 2 * team_size), dtype=torch.int8, device=dev)
        cd = torch.zeros((k, 2 * team_size), dtype=torch.int8, device=dev)
        health = torch.full((self.k, 2 * team_size), 100, dtype=torch.int16, device=dev)
        for idx, agent in enumerate(agents):
            pos[:, idx, 0] = torch.randint(*spawn_y[agent.team], (self.k,), device=dev)  # TODO: figure why warning?
            pos[:, idx, 1] = torch.randint(*spawn_x[agent.team], (self.k,), device=dev)
            ammo[:, idx] = torch.full((self.k,), max_ammo[agent.ptype], device=dev)

        self.bullets = torch.full((k, 2, max_bullets, 3, 2), -1, dtype=torch.half, device=dev)  # (team, type), pos, dir

        for frame in range(game_length_frames):
            states = torch.empty((k, 2 * team_size), dtype=torch.int, device=dev)
            pos_idx = self.cell_idx[pos.unbind(-1)]
            neighbors = self.dist[batch_self_prod(pos_idx).unbind(-1)].view(k, 2 * team_size, 2 * team_size)
            self.bullets[:, :, :, 1, :] += self.bullets[:, :, :, 2, :]

            for idx, agent in enumerate(agents):
                # find visible alive agents
                is_vis_alive = self.is_visible[pos_idx[:, idx].unsqueeze(1), pos_idx].clone()
                is_vis_alive.logical_and_(health != 0)
                is_vis_alive.logical_and_((torch.arange(2 * team_size, device=dev) != idx).unsqueeze(0).expand(k, 2 * team_size))
                neighbors[:, idx, :] *= is_vis_alive
            neighbors[neighbors == 0.] = usable_cells

            for idx, agent in enumerate(agents):
                team = slice(team_size * agent.team, team_size * (agent.team + 1))
                oteam = slice(team_size * (1 - agent.team), team_size * (2 - agent.team))
                # find closest ally
                friend = neighbors[:, idx, team].argmin(dim=1)
                f_ok = neighbors[torch.arange(k, device=dev), idx, friend] != usable_cells
                enemy = neighbors[:, idx, oteam].argmin(dim=1)
                e_ok = neighbors[torch.arange(k, device=dev), idx, enemy] != usable_cells

                # compute friend info
                f_d = neighbors[torch.arange(k, device=dev), idx, friend] >= distance_thresh
                f_a = self.ang8[pos_idx[:, idx], pos_idx[torch.arange(k, device=dev), friend]].div(2, rounding_mode='trunc')
                f_t = friend - (team_size * agent.team)
                f_t += -1 * (f_t > agent.ptype)
                f_h = health[torch.arange(k, device=dev), friend] >= health_thresh
                f_info = f_ok * (f_ok + f_h + 2 * (f_t + 2 * (f_a + 4 * f_d)))

                e_d = neighbors[torch.arange(k, device=dev), idx, enemy] >= distance_thresh
                e_a = self.ang8[pos_idx[:, idx], pos_idx[torch.arange(k, device=dev), enemy]]
                e_t = enemy - (team_size * (1 - agent.team))
                e_info = e_ok * (e_ok + e_t + 3 * (e_a + 8 * e_d))

                s_p = pos_idx[:, idx]
                s_s = 2 * cd[:, idx].bool() + 1 * (~(cd[:, idx].bool()) * ~(ammo[:, idx].bool()))
                s_h = health[:, idx] >= health_thresh

                b_valid = self.bullets[:, 1 - agent.team, :, 0, 0] != -1
                b_pos = self.bullets[:, 1 - agent.team, :, 1, :].long()
                b_pos.clamp_(min=0, max=min(board_height, board_width) - 1)
                b_pos_idx = self.cell_idx[b_pos.unbind(-1)]
                b_vis = self.is_visible[pos_idx[:, idx].unsqueeze(1), b_pos_idx]
                b_ang = torch.atan2(*self.bullets[:, 1 - agent.team, :, 1, :].unbind(-1))
                b_approx = b_ang.isclose(self.ang[pos_idx[:, idx].unsqueeze(1), b_pos_idx], atol=0.25)

                b_valid.logical_and_(b_vis)
                b_valid.logical_and_(b_approx)

                s_b = b_valid * (b_valid + (self.ang8[pos_idx[:, idx].unsqueeze(1), b_pos_idx] / 2))
                s_b = torch.max(s_b, dim=1)[0]

                s_info = s_b + 5 * (s_h + 2 * (s_s + 3 * s_p))
                
                s = f_info + (2 * 2 * 4 * 2) * (e_info + (3 * 8 * 2) * s_info)

                reloading = reload != 0
                dead = health == 0
                s[reloading[:, idx]] = reload_state
                s[dead[:, idx]] = dead_state
                agent.action(s.long(), frame)

            # actions
            for idx, agent in enumerate(agents):
                action = agent.actions[frame]
                move = health[:, idx] != 0
                move.logical_and_(reload[:, idx] == 0)
                new_pos = pos[:, idx] + self.move_delta[action.long()]
                move_ok = self.board[new_pos.unbind(-1)] != 0
                pos[:, idx, :] += move_ok.unsqueeze(1) * self.move_delta[action.long()]

            # actions
            for idx, agent in enumerate(agents):
                action = agent.actions[frame]
                shoot = health[:, idx] != 0
                shoot.logical_and_(reload[:, idx] == 0)
                shoot.logical_and_(cd[:, idx] == 0)
                shoot.logical_and_(ammo[:, idx] != 0)
                shoot.logical_and_(action == 4)
                if agent.ptype == 2:
                    team = slice(team_size * agent.team, team_size * (agent.team + 1))
                    friend = neighbors[:, idx, team].argmin(dim=1)
                    f_ok = neighbors[torch.arange(k, device=dev), idx, friend] != usable_cells
                    shoot.logical_and_(f_ok)
                    heal = shoot * torch.min(
                        torch.tensor(bullet_damage[2], device=dev),
                        100 - health[torch.arange(k, device=dev), friend]
                    )
                    health[torch.arange(k, device=dev), friend] += heal
                    agent.rewards[frame] += heal * damage_reward_ratio
                    cd[:, idx] = shoot * cd_frames[2]
                else:
                    oteam = slice(team_size * (1 - agent.team), team_size * (2 - agent.team))
                    enemy = neighbors[:, idx, oteam].argmin(dim=1)
                    e_ok = neighbors[torch.arange(k, device=dev), idx, enemy] != usable_cells
                    shoot.logical_and_(e_ok)


            #         elif enemy and agent.ptype == 1:
            #             to_shoot.append((enemy, agent.ptype, (agent.y, agent.x)))
            #             agent.ammo += -1
            #             agent.cd = cd[agent.ptype]
            #     elif not agent.ammo:
            #         agent.reload = reload[agent.ptype]
                # if agent.reload == 1:
                #     agent.ammo = max_ammo[agent.ptype]
                # agent.reload = max(0, agent.reload - 1)
                # agent.cd = max(0, agent.cd - 1)

            # for enemy, ptype, (y, x) in to_shoot:
            #     angle = atan2(enemy.y - y, enemy.x - x)
            #     angle += (random.random() * 2 * bullet_spread[agent.ptype]) - bullet_spread[agent.ptype]
            #     bullet = [1 - enemy.team, ptype, [y + 0.5, x + 0.5],
            #               (sin(angle) * bullet_speed[ptype], cos(angle) * bullet_speed[ptype])]
            #     self.bullets.append(bullet)

            # environment response

            # bullets
            # rem_bullets = []
            # for bullet in self.bullets:
            #     hit = False
            #
            #     for agent in agents:
            #         if not agent.health or agent.team == bullet[0]:
            #             continue
            #         if dist((agent.y + 0.5, agent.x + 0.5), bullet[2]) <= adjusted_agent_radius:
            #             hit = True
            #             dmg = min(bullet_damage[bullet[1]], agent.health)
            #             rewards[3 * bullet[0] + bullet[1]] += dmg * damage_reward_ratio
            #             rewards[3 * agent.team + agent.ptype] -= dmg * damage_reward_ratio
            #             agent.health -= dmg
            #             if not agent.health:
            #                 rewards[3 * bullet[0] + bullet[1]] += kill_reward
            #                 rewards[3 * agent.team + agent.ptype] -= kill_reward
            #
            #     if not hit and self.ok(int(bullet[2][0]), int(bullet[2][1])):
            #         if self.board[int(bullet[2][0])][int(bullet[2][1])] != 0:
            #             rem_bullets.append(bullet)
            #
            # self.bullets = rem_bullets

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
                        pygame.draw.line(self.screen, (255 * (idx == 0), 50, 255),
                                         *((point[0] * cell_size, point[1] * cell_size) for point in line))

                    for idx, line in enumerate(self.olines):
                        pygame.draw.line(self.screen, (255, 50, 255 * (idx == 0)),
                                         *((point[0] * cell_size, point[1] * cell_size) for point in line))
                    break

            pygame.display.flip()
            time.sleep(1 / (iframes * fps))


simul = Simulator(batch_size)
simul.show_game = False

# simul.show_game = True
# simul.episode()
# simul.show_game = False

for i_ in range(1000000):
    for team_ in simul.teams:
        for agent_ in team_:
            agent_.epsilon *= 0.99
            agent_.alpha *= 0.99

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

    # simul.show_game = True
    # simul.episode()
    # simul.show_game = False

pygame.quit()
