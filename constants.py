cell_size = 26
board_width, board_height = 45, 34
width, height = cell_size * board_width, cell_size * board_height

spawn_y = ((29, 32), (7, 10))
spawn_x = ((6, 9), (40, 43))

capture_y = ((6, 11), (22, 25))
capture_x = ((5, 9), (13, 17))

game_length_s = 45
fps = 30
game_length_frames = game_length_s * fps
iframes = 4

distance_thresh = 8
health_thresh = 51

adjusted_agent_radius = 2 ** -0.5
bullet_speed = [2.4, 1.8]
bullet_range = [10000, 11]
bullet_spread = [0, 0.0872665]  # 5 deg in radians
bullet_damage = [0.4, 0.08, 0.4]
max_ammo = [2, 20, 10000]
cd = [fps, fps // 4, fps * 3]
reload = [fps * 4, fps * 4, 0]

capture_thresh = fps * 10
