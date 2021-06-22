import math
from constants import *
from dataclasses import dataclass


def intersect(l1, l2):
    # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line_segment
    y1, x1, y2, x2 = l1
    y3, x3, y4, x4 = l2
    num1 = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    num2 = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den != 0:
        math.int = (y1 + (num1 / den) * (y2 - y1), x1 + (num1 / den) * (x2 - x1))
    return den != 0 and 0 <= num1 / den <= 1 and 0 <= num2 / den <= 1


def dist(p1, p2):
    return (((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2)) ** 0.5


def round_ang(p1, p2, k):
    return round(k * math.atan2(p2[0] - p1[0], p2[1] - p1[1]) / (2 * math.pi)) % k


def friend_state(player, oplayer):
    if not oplayer:
        return ()
    td = dist((player.y, player.x), (oplayer.y, oplayer.x)) >= distance_thresh
    ta = round_ang((player.y, player.x), (oplayer.y, oplayer.x), 4)
    tt = oplayer.ptype - (oplayer.ptype > player.ptype)
    th = oplayer.health >= health_thresh
    return ((td * 4 + ta) * 2 + tt) * 2 + th,


def enemy_state(player, oplayer):
    if not oplayer:
        return ()
    ed = dist((player.y, player.x), (oplayer.y, oplayer.x)) >= distance_thresh
    ea = round_ang((player.y, player.x), (oplayer.y, oplayer.x), 8)
    et = oplayer.ptype
    return (ed * 8 + ea) * 3 + et,


def add_tuple(t1, t2):
    return tuple(e1 + e2 for e1, e2 in zip(t1, t2))
