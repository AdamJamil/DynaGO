cell_size = 40  # 26
board_width, board_height = 10, 10  # 45, 34
width, height = cell_size * board_width, cell_size * board_height
usable_cells = 56 + 8

team_size = 3
team_types = [1, 1, 1]

spawn_y = ((7, 8), (1, 2))  # ((29, 32), (7, 10))
spawn_x = ((1, 2), (7, 8))  # ((6, 9), (40, 43))

capture_y = ((600, 11), (22, 25))
capture_x = ((500, 9), (13, 17))

game_length_s = 100
ifps = 8
game_length_frames = game_length_s * ifps
fps = 120
iframes = 1

distance_thresh = 8
health_thresh = 51

adjusted_agent_radius = 2 ** -0.5
bullet_speed = [1.2, 0.9]
bullet_range = [10000, 11]
bullet_spread = [0, 0.0872665]  # 5 deg in radians
bullet_damage = [40, 8, 40]
max_ammo = [2, 20, 10000]
cd = [ifps, ifps // 4, ifps * 3]
reload = [int(ifps * 0.8), int(ifps * 0.8), 0]

capture_thresh = ifps * 10

win_reward = 0
tie_penalty = 0
damage_reward_ratio = 0.01
kill_reward = 0.0

self_info_size = usable_cells * 3 * 2 * 5
friend_info_size = 2 * 4 * 2 * 2
enemy_info_size = 2 * 8 * 3
