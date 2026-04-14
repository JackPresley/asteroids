# minimax_asteroids.py
#
# Adds an AI mode (minimax) to the asteroids game without modifying asteroids.py.
#
import pygame, sys
from pygame.locals import *
from math import sin, cos, pi, sqrt, atan2
from random import randint

import asteroids as _astro

Space       = _astro.Space
Ship        = _astro.Ship
Bullet      = _astro.Bullet
BigRock     = _astro.BigRock
MediumRock  = _astro.MediumRock
SmallRock   = _astro.SmallRock
Score       = _astro.Score
Fader       = _astro.Fader
Burst       = _astro.Burst
textBlit    = _astro.textBlit
NUM_ROCKS   = _astro.NUM_ROCKS
WIDTH       = _astro.WIDTH
HEIGHT      = _astro.HEIGHT
winWidth    = _astro.winWidth
winHeight   = _astro.winHeight
FPS         = _astro.FPS
REFERENCE_FPS = _astro.REFERENCE_FPS
WHITE       = _astro.WHITE
GREY        = _astro.GREY
BLACK       = _astro.BLACK
BLUE        = _astro.BLUE
RED         = _astro.RED

# ---------------------------------------------------------------------------
# Physics scaling
# ---------------------------------------------------------------------------

SIM_DT = REFERENCE_FPS / FPS  # ~2.5: all step() calls use this
_DIAG = sqrt((winWidth / 2) ** 2 + (winHeight / 2) ** 2)

_SIN8 = [sin(i * pi / 4) for i in range(8)]
_COS8 = [cos(i * pi / 4) for i in range(8)]
CPA_HORIZON = 60  # frames to look ahead for closest-point-of-approach

# ---------------------------------------------------------------------------
# Toroidal distance helpers
# ---------------------------------------------------------------------------

def wrap_dist(a, b, mod):
    d = abs(a - b)
    return min(d, mod - d)

def wrap_delta(a, b, mod):
    """Signed shortest-path delta from a to b on a toroidal axis."""
    d = b - a
    if d > mod / 2:
        d -= mod
    elif d < -mod / 2:
        d += mod
    return d

def torus_dist(x1, y1, x2, y2):
    dx = wrap_dist(x1, x2, winWidth)
    dy = wrap_dist(y1, y2, winHeight)
    return sqrt(dx * dx + dy * dy)

def intercept_aim_angle(ship, rock):
    """Return signed angle (deg) from ship heading to intercept point of rock."""
    BULLET_SPEED = 5.0
    px = wrap_delta(ship.x, rock.x, winWidth)
    py = wrap_delta(ship.y, rock.y, winHeight)
    vx = rock.dx - ship.dx
    vy = rock.dy - ship.dy
    a = vx * vx + vy * vy - BULLET_SPEED ** 2
    b = 2.0 * (px * vx + py * vy)
    c = px * px + py * py
    t = None
    if abs(a) < 1e-9:
        if abs(b) > 1e-9:
            tc = -c / b
            if tc > 0:
                t = tc
    else:
        disc = b * b - 4.0 * a * c
        if disc >= 0:
            sd = sqrt(disc)
            for tc in ((-b - sd) / (2 * a), (-b + sd) / (2 * a)):
                if tc > 0 and (t is None or tc < t):
                    t = tc
    if t is not None:
        ix, iy = px + vx * t, py + vy * t
        target_angle = atan2(-ix, -iy) * 180 / pi
    else:
        target_angle = atan2(-px, -py) * 180 / pi
    return (target_angle - ship.theta + 180) % 360 - 180

# ---------------------------------------------------------------------------
# Lightweight state for simulation (no pygame sprites)
# ---------------------------------------------------------------------------

class SimShip:
    __slots__ = ('x', 'y', 'dx', 'dy', 'theta', 'accel', 'd_theta')

    def __init__(self, x, y, dx, dy, theta, accel=0.02):
        self.x = x % winWidth
        self.y = y % winHeight
        self.dx = dx
        self.dy = dy
        self.theta = theta
        self.accel = accel
        self.d_theta = 0.0

    def copy(self):
        s = SimShip(self.x, self.y, self.dx, self.dy, self.theta, self.accel)
        s.d_theta = self.d_theta
        return s

    def step(self, thrust, left, right, dt=SIM_DT):
        self.d_theta = -1.5 if right else (1.5 if left else 0.0)
        if thrust:
            self.dx += self.accel * dt * -sin(self.theta * pi / 180)
            self.dy += self.accel * dt * -cos(self.theta * pi / 180)
        self.theta += self.d_theta * dt
        self.x = (self.x + self.dx * dt) % winWidth
        self.y = (self.y + self.dy * dt) % winHeight


class SimRock:
    __slots__ = ('x', 'y', 'dx', 'dy', 'radius')

    def __init__(self, x, y, dx, dy, radius):
        self.x = x % winWidth
        self.y = y % winHeight
        self.dx = dx
        self.dy = dy
        self.radius = radius

    def copy(self):
        return SimRock(self.x, self.y, self.dx, self.dy, self.radius)

    def step(self, dt=SIM_DT):
        self.x = (self.x + self.dx * dt) % winWidth
        self.y = (self.y + self.dy * dt) % winHeight


class SimBullet:
    __slots__ = ('x', 'y', 'dx', 'dy', 'dist_left', 'speed')

    def __init__(self, x, y, dx, dy, dist_left):
        self.x = x % winWidth
        self.y = y % winHeight
        self.dx = dx
        self.dy = dy
        self.dist_left = dist_left
        self.speed = sqrt(dx * dx + dy * dy)

    def copy(self):
        return SimBullet(self.x, self.y, self.dx, self.dy, self.dist_left)

    def step(self, dt=SIM_DT):
        self.x = (self.x + self.dx * dt) % winWidth
        self.y = (self.y + self.dy * dt) % winHeight
        self.dist_left -= self.speed * dt


def make_sim_bullet(ship_sim):
    speed = 5
    tdx = -sin(ship_sim.theta * pi / 180)
    tdy = -cos(ship_sim.theta * pi / 180)
    # Spin correction: bullet tip sweeps sideways proportional to d_theta
    bdx = speed * tdx + ship_sim.d_theta * tdy + ship_sim.dx
    bdy = speed * tdy - ship_sim.d_theta * tdx + ship_sim.dy
    return SimBullet(ship_sim.x, ship_sim.y, bdx, bdy, 6 * winHeight / 7)

# ---------------------------------------------------------------------------
# Minimax AI
# ---------------------------------------------------------------------------

# Actions: (thrust, left, right, shoot)
ACTIONS = [
    (False, False, False, False),  # drift
    (True,  False, False, False),  # thrust
    (False, True,  False, False),  # left
    (False, False, True,  False),  # right
    (True,  True,  False, False),  # thrust+left
    (True,  False, True,  False),  # thrust+right
    (False, False, False, True),   # shoot
    (False, True,  False, True),   # left+shoot
    (False, False, True,  True),   # right+shoot
    (True,  False, False, True),   # thrust+shoot
]

SHIP_RADIUS = 18  # approximate collision radius
SIM_STEPS_PER_ACTION = 4   # frames per action choice (finer evasion granularity)
DEPTH = 3


def evaluate(ship, rocks, bullets):
    """Heuristic evaluation of a game state. Higher is better."""
    if not rocks:
        # Wave cleared. Guide the search toward a good starting position for
        # the next wave: centered, slow, and oriented to brake.
        fwd_x = -sin(ship.theta * pi / 180)
        fwd_y = -cos(ship.theta * pi / 180)
        speed = sqrt(ship.dx ** 2 + ship.dy ** 2)
        cx, cy = winWidth / 2, winHeight / 2
        center_dist = sqrt((ship.x - cx) ** 2 + (ship.y - cy) ** 2)
        max_r = min(winWidth, winHeight) / 2
        center_bonus  = 1000 * max(0.0, 1.0 - center_dist / max_r)
        speed_penalty = max(0.0, speed - 1.0) ** 2 * 400
        brake_bonus   = 0.0
        if speed > 0.8:
            brake_dot   = -(fwd_x * ship.dx + fwd_y * ship.dy) / speed
            brake_bonus = max(0.0, brake_dot) * 400
        return 10000 + center_bonus - speed_penalty + brake_bonus

    score = 0.0
    fwd_x = -sin(ship.theta * pi / 180)
    fwd_y = -cos(ship.theta * pi / 180)
    speed = sqrt(ship.dx ** 2 + ship.dy ** 2)

    # ---- Build rock cache + collision check + CPA danger ----
    # Cache (r, dx_to, dy_to, center_dist, edge_dist, t_cpa, cpa_dist) for reuse below.
    total_danger = 0.0
    min_edge_dist = float('inf')
    rock_info = []

    for r in rocks:
        dx_to = wrap_delta(ship.x, r.x, winWidth)
        dy_to = wrap_delta(ship.y, r.y, winHeight)
        center_dist = sqrt(dx_to * dx_to + dy_to * dy_to)
        edge_dist = center_dist - r.radius - SHIP_RADIUS

        if edge_dist < 0:
            return -100000  # collision

        if edge_dist < min_edge_dist:
            min_edge_dist = edge_dist

        rel_vx = (r.dx - ship.dx) * SIM_DT
        rel_vy = (r.dy - ship.dy) * SIM_DT
        rel_speed_sq = rel_vx * rel_vx + rel_vy * rel_vy
        if rel_speed_sq > 0.001:
            t_cpa = -(dx_to * rel_vx + dy_to * rel_vy) / rel_speed_sq
            t_cpa = max(0.0, min(t_cpa, float(CPA_HORIZON)))
            cpa_dx = dx_to + rel_vx * t_cpa
            cpa_dy = dy_to + rel_vy * t_cpa
            cpa_dist = sqrt(cpa_dx * cpa_dx + cpa_dy * cpa_dy) - r.radius * 0.7 - SHIP_RADIUS
        else:
            t_cpa = float(CPA_HORIZON)
            cpa_dist = edge_dist

        rock_info.append((r, dx_to, dy_to, center_dist, edge_dist, t_cpa, cpa_dist))

        # CPA danger: hyperbolic, blindside-boosted, urgency-scaled.
        # At cpa_dist < ~7 danger exceeds the aim bonus; evasion takes priority.
        if cpa_dist < 200:
            size_mult = r.radius / 30.0
            aim_dot = (fwd_x * dx_to + fwd_y * dy_to) / max(1.0, center_dist)
            blindside_mult = 1.0 if aim_dot > 0.3 else (1.8 - aim_dot)
            urgency = 1.0 + max(0.0, 1.0 - t_cpa / 20.0)
            total_danger += size_mult * blindside_mult * urgency / max(1.0, cpa_dist + 8.0)

    score -= total_danger * 1500.0

    # Safe-distance baseline: clear contrast between safe and dangerous states.
    score += min(min_edge_dist, 300)

    # ---- Escape corridor scoring (8 rays from the ship) ----
    # Replaces multi-rock convergence penalty with geometry-aware corridor check:
    # being boxed in on all sides scores badly even if no single rock is close.
    corridor_dists = []
    for i in range(8):
        ray_dx, ray_dy = _SIN8[i], _COS8[i]
        min_clear = 400.0
        for r, dx_to, dy_to, center_dist, edge_dist, t_cpa, cpa_dist in rock_info:
            t = dx_to * ray_dx + dy_to * ray_dy
            if t < 0:
                continue  # rock is behind this ray
            perp = abs(dx_to * ray_dy - dy_to * ray_dx)
            if perp < r.radius + SHIP_RADIUS:
                clear = max(0.0, t - r.radius - SHIP_RADIUS)
                if clear < min_clear:
                    min_clear = clear
        corridor_dists.append(min_clear)

    corridor_dists.sort()
    best_escape  = corridor_dists[-1]
    second_escape = corridor_dists[-2]
    if best_escape < 100:
        score -= 600.0
    elif best_escape < 200:
        score -= 250.0 * (1.0 - best_escape / 200.0)
    score += min(second_escape, 200) * 0.5  # bonus for having multiple open routes

    # ---- Bullet scoring ----
    # Bullet-spawn danger: if a bullet is about to destroy a big/medium rock near
    # the ship, the children will spawn right there — reward the position, not the kill.
    for b in bullets:
        if b.dist_left <= 0:
            continue
        for r, dx_to, dy_to, center_dist, edge_dist, t_cpa, cpa_dist in rock_info:
            if r.radius < 30:
                continue
            bdx = wrap_delta(b.x, r.x, winWidth)
            bdy = wrap_delta(b.y, r.y, winHeight)
            bd = sqrt(bdx * bdx + bdy * bdy)
            if bd < r.radius + 20 and edge_dist < 130:
                score -= 400.0 * (1.0 - edge_dist / 130.0)

    # Bullet-rock CPA: reward bullets geometrically on track to hit a rock.
    for b in bullets:
        if b.dist_left <= 0:
            continue
        for r, dx_to, dy_to, center_dist, edge_dist, t_cpa, cpa_dist in rock_info:
            bdx = wrap_delta(b.x, r.x, winWidth)
            bdy = wrap_delta(b.y, r.y, winHeight)
            brvx = (b.dx - r.dx) * SIM_DT
            brvy = (b.dy - r.dy) * SIM_DT
            br_speed_sq = brvx * brvx + brvy * brvy
            if br_speed_sq < 0.001:
                continue
            t_hit = (bdx * brvx + bdy * brvy) / br_speed_sq
            if t_hit < 0:
                continue  # bullet already past CPA, moving away
            cpa_bx = bdx - brvx * t_hit
            cpa_by = bdy - brvy * t_hit
            if sqrt(cpa_bx * cpa_bx + cpa_by * cpa_by) < r.radius + 8:
                safe_mult = 1.0 if r.radius < 30 or edge_dist >= 130 else 0.2
                score += 150.0 * safe_mult

    # ---- Speed management ----
    # Station-keeping bonus: explicit positive reward for being nearly stopped.
    # At short search horizons the agent can't see far-future consequences of speed,
    # so we encode "slow = safe" directly rather than relying solely on CPA penalties.
    if speed < 1.5:
        score += 150.0 * max(0.0, 1.0 - speed / 1.5)

    # Rock-count-scaled speed penalty: threshold lowered to 0.5 so even moderate
    # speeds are penalised.  Being at speed 1.5 through 3 rocks costs ~285 — more
    # than the vel_toward_safe bonus can compensate.
    rock_count_mult = 1.0 + min(len(rocks), 8) * 0.3
    if speed > 0.5:
        score -= (speed - 0.5) ** 2 * 150.0 * rock_count_mult

    # ---- Evasion mode: suppress all positioning when any rock is on a CPA approach ----
    # With a short search horizon, positioning rewards (safe zone, centre return) can
    # outweigh the incremental danger signal and pull the agent toward rocks.
    # When under any CPA threat the ship should focus purely on surviving.
    evasion_mode = any(cpa_dist < 120
                       for _, _, _, _, _, _, cpa_dist in rock_info)

    # ---- Retrograde thrust (braking orientation reward) ----
    # Always active: at any speed above 1.0 braking orientation is among the
    # highest-value choices, competing with danger avoidance not positioning.
    if speed > 0.1:
        retro_dot = -(fwd_x * (ship.dx / speed) + fwd_y * (ship.dy / speed))
        if retro_dot > 0:
            score += 500 * retro_dot * min(1.0, speed / 3.0)

    if not evasion_mode:
        # ---- Aim ----
        # Skip rocks already tracked by a closing bullet (intercept_aim_angle is
        # the exact quadratic solution, more accurate than the linear DQN approximation).
        targeted = set()
        for b in bullets:
            if rocks:
                closest_r = min(range(len(rocks)),
                                key=lambda i: torus_dist(b.x, b.y, rocks[i].x, rocks[i].y))
                bd = torus_dist(b.x, b.y, rocks[closest_r].x, rocks[closest_r].y)
                closing = -(b.dx * wrap_delta(b.x, rocks[closest_r].x, winWidth) +
                            b.dy * wrap_delta(b.y, rocks[closest_r].y, winHeight)) / max(bd, 1e-6)
                if closing > 0:
                    targeted.add(closest_r)
        untargeted = [r for i, r in enumerate(rocks) if i not in targeted] or rocks
        nearest = min(untargeted, key=lambda r: torus_dist(ship.x, ship.y, r.x, r.y))
        # Continuous falloff so the search has a real gradient toward better aim:
        # each degree of improvement registers in the score rather than snapping
        # between discrete bands.  Perfect aim = +150, 45° off = 0.
        aim_diff = abs(intercept_aim_angle(ship, nearest))
        if aim_diff < 45:
            score += 150 * max(0.0, 1.0 - aim_diff / 45.0)

        # ---- Global open-space positioning ----
        # 4×3 grid: find safest cell, reward proximity and gentle drift toward it.
        best_d = -1.0
        best_x, best_y = winWidth / 2, winHeight / 2
        for gi in range(4):
            for gj in range(3):
                gx = (gi + 0.5) * winWidth / 4
                gy = (gj + 0.5) * winHeight / 3
                gd = min(torus_dist(gx, gy, r.x, r.y) for r in rocks)
                if gd > best_d:
                    best_d, best_x, best_y = gd, gx, gy
        score += 400 * max(0.0, 1.0 - torus_dist(ship.x, ship.y, best_x, best_y) / (_DIAG * 0.4))

        dist_to_safe = torus_dist(ship.x, ship.y, best_x, best_y)
        if dist_to_safe > 80 and speed > 0.05:
            sdx = wrap_delta(ship.x, best_x, winWidth)
            sdy = wrap_delta(ship.y, best_y, winHeight)
            safe_ux = sdx / dist_to_safe
            safe_uy = sdy / dist_to_safe
            vel_toward = (ship.dx * safe_ux + ship.dy * safe_uy) / speed
            if vel_toward > 0:
                score += 50 * vel_toward * min(1.0, dist_to_safe / 200)

        # ---- Centre return (graded by rock count) ----
        cx, cy = winWidth / 2, winHeight / 2
        dist_centre = torus_dist(ship.x, ship.y, cx, cy)
        centred = max(0.0, 1.0 - dist_centre / (_DIAG * 0.35))
        if len(rocks) == 1:
            score += 800 * centred
            score += 500 * max(0.0, 1.0 - speed / 0.5) * centred
            if bullets:
                score += 400 * centred  # reward having a bullet in-flight while centred
        elif len(rocks) == 2:
            score += 500 * centred
        elif len(rocks) == 3:
            score += 200 * centred

    return score


def _spawn_children(rock):
    """Return deterministic child SimRocks for minimax simulation.

    The real game randomises child velocities; we use a fixed perpendicular
    split so the simulation is consistent across search nodes.  Children
    scatter perpendicular to the parent's travel direction (or along the x
    axis if the parent is stationary), which keeps them from immediately
    overlapping and gives a representative worst-case spread.
    """
    children = []
    if rock.radius == 50:  # BigRock -> 2 MediumRock at avg speed ~0.5/axis
        spd = 0.5
        mag = sqrt(rock.dx ** 2 + rock.dy ** 2)
        if mag < 1e-6:
            px, py = 1.0, 0.0
        else:
            px, py = -rock.dy / mag, rock.dx / mag  # unit perpendicular
        children.append(SimRock(rock.x, rock.y, px * spd, py * spd, 30))
        children.append(SimRock(rock.x, rock.y, -px * spd, -py * spd, 30))
    elif rock.radius == 30:  # MediumRock -> 2 SmallRock at avg speed ~0.7/axis
        spd = 0.7
        mag = sqrt(rock.dx ** 2 + rock.dy ** 2)
        if mag < 1e-6:
            px, py = 1.0, 0.0
        else:
            px, py = -rock.dy / mag, rock.dx / mag
        children.append(SimRock(rock.x, rock.y, px * spd, py * spd, 15))
        children.append(SimRock(rock.x, rock.y, -px * spd, -py * spd, 15))
    # SmallRock -> nothing
    return children


def sim_step(ship, rocks, bullets, action):
    """Simulate SIM_STEPS_PER_ACTION frames with the given action. Mutates copies."""
    thrust, left, right, shoot = action
    new_ship = ship.copy()
    new_rocks = [r.copy() for r in rocks]
    new_bullets = [b.copy() for b in bullets]

    if shoot:
        new_bullets.append(make_sim_bullet(new_ship))

    for _ in range(SIM_STEPS_PER_ACTION):
        new_ship.step(thrust, left, right)
        for r in new_rocks:
            r.step()
        remaining_bullets = []
        for b in new_bullets:
            b.step()
            if b.dist_left > 0:
                remaining_bullets.append(b)
        new_bullets = remaining_bullets

        # Check bullet-rock collisions
        hit_rocks = set()
        hit_bullets = set()
        for bi, b in enumerate(new_bullets):
            for ri, r in enumerate(new_rocks):
                if ri not in hit_rocks and torus_dist(b.x, b.y, r.x, r.y) < r.radius + 5:
                    hit_rocks.add(ri)
                    hit_bullets.add(bi)
                    break
        if hit_rocks:
            spawned = []
            for ri in hit_rocks:
                spawned.extend(_spawn_children(new_rocks[ri]))
            new_rocks = [r for i, r in enumerate(new_rocks) if i not in hit_rocks] + spawned
            new_bullets = [b for i, b in enumerate(new_bullets) if i not in hit_bullets]

        # Check ship-rock collision
        for r in new_rocks:
            if torus_dist(new_ship.x, new_ship.y, r.x, r.y) < r.radius * 0.7 + SHIP_RADIUS:
                return new_ship, new_rocks, new_bullets, True  # dead

    return new_ship, new_rocks, new_bullets, False


def _prune_actions(ship, rocks, bullets):
    """Return a reduced list of action indices based on current threat level.

    In imminent danger: movement only (6 actions, no shooting).
    In moderate danger: movement + turn-while-shoot (8 actions).
    When safe: rotate/drift/shoot (4-7 actions).
    Reduces branching factor from 10 to ~6, enabling DEPTH=4 within budget.
    """
    if not rocks:
        return [0]   # wave cleared — drift

    speed = sqrt(ship.dx ** 2 + ship.dy ** 2)
    danger   = speed > 2.0
    imminent = False   # rock about to hit — suppress shooting

    for r in rocks:
        dx = wrap_delta(ship.x, r.x, winWidth)
        dy = wrap_delta(ship.y, r.y, winHeight)
        dist = sqrt(dx * dx + dy * dy)
        edge = dist - r.radius - SHIP_RADIUS
        if edge < 120:
            danger = True
        if edge < 55:
            imminent = True
            break
        rvx = (r.dx - ship.dx) * SIM_DT
        rvy = (r.dy - ship.dy) * SIM_DT
        rs2 = rvx * rvx + rvy * rvy
        if rs2 > 0.001:
            t = max(0.0, min(-(dx * rvx + dy * rvy) / rs2, 60.0))
            cpx = dx + rvx * t
            cpy = dy + rvy * t
            cpa = sqrt(cpx * cpx + cpy * cpy) - r.radius * 0.7 - SHIP_RADIUS
            if cpa < 90:
                danger = True
            if cpa < 50 and t < 25:
                imminent = True
        if imminent:
            break

    if imminent:
        return [0, 1, 2, 3, 4, 5]           # evasion only, no shoot
    elif danger:
        return [0, 1, 2, 3, 4, 5, 7, 8]     # evade + turn-while-shooting
    else:
        actions = [0, 2, 3, 6, 7, 8]         # safe: drift, rotate, shoot variants
        if speed > 0.8:
            actions.append(1)                 # allow thrust for braking
        return actions


def _survival_probe(ship, rocks, horizon=16):
    """Drift the ship for `horizon` physics frames and detect imminent collision.

    Returns a large penalty if the current velocity vector leads to death,
    zero otherwise.  Cheap: no branching, no object allocation — runs in
    O(horizon × len(rocks)) time.
    """
    if not rocks:
        return 0
    sx, sy   = ship.x, ship.y
    sdx, sdy = ship.dx, ship.dy
    # Store rock state as plain tuples: (x, y, dx, dy, kill_radius)
    rdata = [(r.x, r.y, r.dx, r.dy, r.radius * 0.7 + SHIP_RADIUS) for r in rocks]
    hw, hh = winWidth / 2, winHeight / 2
    for _ in range(horizon):
        sx = (sx + sdx * SIM_DT) % winWidth
        sy = (sy + sdy * SIM_DT) % winHeight
        next_rdata = []
        for rx, ry, rdx, rdy, kr in rdata:
            rx = (rx + rdx * SIM_DT) % winWidth
            ry = (ry + rdy * SIM_DT) % winHeight
            next_rdata.append((rx, ry, rdx, rdy, kr))
            ddx = sx - rx
            if   ddx >  hw: ddx -= winWidth
            elif ddx < -hw: ddx += winWidth
            ddy = sy - ry
            if   ddy >  hh: ddy -= winHeight
            elif ddy < -hh: ddy += winHeight
            if ddx * ddx + ddy * ddy < kr * kr:
                return -30000
        rdata = next_rdata
    return 0


def _cheap_sort_key(ship, rocks):
    """O(N) CPA-min heuristic for move ordering at non-leaf nodes.

    Returns a scalar: higher = better state.  Much cheaper than evaluate()
    since it skips corridor scoring, bullet tracking, and aim computation.
    """
    if not rocks:
        return 10000
    speed = sqrt(ship.dx ** 2 + ship.dy ** 2)
    min_cpa = float('inf')
    hw, hh = winWidth / 2, winHeight / 2
    for r in rocks:
        dx = ship.x - r.x
        if   dx >  hw: dx -= winWidth
        elif dx < -hw: dx += winWidth
        dy = ship.y - r.y
        if   dy >  hh: dy -= winHeight
        elif dy < -hh: dy += winHeight
        dist = sqrt(dx * dx + dy * dy)
        edge = dist - r.radius - SHIP_RADIUS
        if edge < 0:
            return -100000
        rel_vx = (r.dx - ship.dx) * SIM_DT
        rel_vy = (r.dy - ship.dy) * SIM_DT
        rs2 = rel_vx * rel_vx + rel_vy * rel_vy
        if rs2 > 0.001:
            t = max(0.0, min(-(dx * rel_vx + dy * rel_vy) / rs2, 60.0))
            cpx = dx + rel_vx * t
            cpy = dy + rel_vy * t
            cpa = sqrt(cpx * cpx + cpy * cpy) - r.radius * 0.7 - SHIP_RADIUS
        else:
            cpa = edge
        if cpa < min_cpa:
            min_cpa = cpa
    return min_cpa - speed * 50.0


def minimax(ship, rocks, bullets, depth, bound=float('inf')):
    """Return (best_score, best_action_index).

    Key design choices vs. the original:
    - _prune_actions reduces branching factor from 10 to ~6 at every depth,
      making DEPTH=4 feasible without a time budget.
    - Children are simulated once and cached; the same result is used for
      both move ordering and the recursive search (no double simulation).
    - _survival_probe is applied at leaves: if the ship's current velocity
      leads to collision within 24 drift frames, the state is penalised heavily.
    - Alpha-beta bound propagated from parent ensures early cutoffs.
    """
    if depth == 0 or not rocks:
        return evaluate(ship, rocks, bullets) + _survival_probe(ship, rocks), 0

    # Simulate every candidate action once; cache results for ordering + search.
    action_indices = _prune_actions(ship, rocks, bullets)
    children = []
    for ai in action_indices:
        ns, nr, nb, dead = sim_step(ship, rocks, bullets, ACTIONS[ai])
        if dead:
            quick = -100000
        elif depth == 1:
            quick = evaluate(ns, nr, nb)      # full eval: used as leaf score
        else:
            quick = _cheap_sort_key(ns, nr)   # cheap: used for ordering only
        children.append((quick, ai, ns, nr, nb, dead))
    children.sort(reverse=True, key=lambda c: c[0])   # best-first for alpha-beta

    best_score  = float('-inf')
    best_action = children[0][1] if children else 0

    for quick, ai, ns, nr, nb, dead in children:
        if dead:
            action_score = -100000
        elif depth == 1:
            # Leaf: cached eval + survival probe (no further recursion).
            action_score = quick + _survival_probe(ns, nr)
        else:
            action_score, _ = minimax(ns, nr, nb, depth - 1, best_score)

        if action_score > best_score:
            best_score  = action_score
            best_action = ai
            if best_score >= bound:
                break   # alpha-beta cut

    return best_score, best_action

# ---------------------------------------------------------------------------
# Build the simulation state from live game objects
# ---------------------------------------------------------------------------

def build_sim_state(ship, rocks, bullets):
    s = SimShip(ship.p.x, ship.p.y, ship.dx, ship.dy, ship._theta)
    s.d_theta = ship.d_theta
    rs = []
    for r in rocks:
        radius = 50  # default
        if isinstance(r, SmallRock):
            radius = 15
        elif isinstance(r, MediumRock):
            radius = 30
        elif isinstance(r, BigRock):
            radius = 50
        rs.append(SimRock(r.p.x, r.p.y, r.dx, r.dy, radius))
    bs = []
    for b in bullets:
        remaining = b.distance - b.distance_travelled
        bs.append(SimBullet(b.p.x, b.p.y, b.dx, b.dy, remaining))
    return s, rs, bs

# ---------------------------------------------------------------------------
# Throttled AI: only recompute every N frames
# ---------------------------------------------------------------------------

class AIController:
    def __init__(self):
        self.timer = 0
        self.action = ACTIONS[0]
        self.recompute_interval = SIM_STEPS_PER_ACTION  # in reference frames
        self.shoot_cooldown = 0
        self.first = True

    def get_action(self, ship, rocks, bullets, dt=1):
        self.timer += dt
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= dt

        if self.first or self.timer >= self.recompute_interval:
            self.first = False
            self.timer = 0
            s, rs, bs = build_sim_state(ship, rocks, bullets)
            _, best = minimax(s, rs, bs, DEPTH)
            self.action = ACTIONS[best]

            # enforce shoot cooldown
            if self.action[3] and self.shoot_cooldown > 0:
                self.action = (self.action[0], self.action[1], self.action[2], False)
            elif self.action[3]:
                self.shoot_cooldown = 20  # reference frames between shots (matches Burst delay)

        return self.action

# ---------------------------------------------------------------------------
# Modified main with AI mode option on splash screen
# ---------------------------------------------------------------------------

def main():

    global fpsClock

    pygame.init()
    fpsClock = pygame.time.Clock()
    _astro.fpsClock = fpsClock  # Fader methods reference fpsClock from the asteroids module
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Asteroids')
    pygame.mouse.set_visible(0)

    fader = Fader(screen)

    # --- modified start_up with AI option ---
    ai_mode = start_up_with_ai(fader, screen)

    ship = Ship(screen)
    bullets = pygame.sprite.Group()
    rocks = pygame.sprite.Group()
    burst = Burst()

    starting_up = True
    pause = False

    while len(rocks) < NUM_ROCKS:
        BigRock(screen, rocks)

    num_rocks = NUM_ROCKS

    ai = AIController() if ai_mode else None

    while True:
        dt = fpsClock.tick(FPS) * REFERENCE_FPS / 1000.0
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                if not ai_mode:
                    if event.key == K_b:
                        burst.shoot = True
                    if event.key == K_SPACE:
                        Bullet(screen, ship, bullets)
            elif event.type == KEYUP:
                if event.key == K_q:
                    pygame.quit()
                    sys.exit()
                if event.key == K_p and not ai_mode:
                    pause = True

        if ai_mode:
            action = ai.get_action(ship, rocks, bullets, dt)
            thrust, left, right, shoot = action
        else:
            shoot = False
            thrust = left = right = False
            keys = pygame.key.get_pressed()
            if keys[K_UP]:
                thrust = True
            if keys[K_LEFT]:
                left = True
            if keys[K_RIGHT]:
                right = True

        screen.fill(WHITE)

        ship.update(thrust, left, right, dt)  # sets _theta_dx/_theta_dy

        if ai_mode:
            if shoot:
                Bullet(screen, ship, bullets)
        elif burst.update(dt):
            Bullet(screen, ship, bullets)

        if bullets:
            bullets.update(bullets, rocks, dt)
        rocks.update(dt)

        if pygame.sprite.spritecollideany(ship, rocks, pygame.sprite.collide_circle_ratio(.7)):
            fader.use_a_life()
            fader.reset()
            if Score.getLives() == 0:
                fader.lose()
                num_rocks = NUM_ROCKS
                Score.reset()
            bullets.empty()
            rocks.empty()
            del ship
            pygame.event.clear()
            ship = Ship(screen)
            while len(rocks) < num_rocks:
                BigRock(screen, rocks)
            if ai_mode:
                ai = AIController()

        Score.draw(screen, rocks)

        if ai_mode:
            textBlit(screen, "AI MODE (minimax)", "Arial", 30, BLUE,
                     "bottomleft", winWidth / 20, 18 * winHeight / 20, False)

        if len(rocks) == 0:
            if fader.frames > 0:
                fader.lifeBonus(dt)
            else:
                fader.reset()
                bullets.empty()
                pygame.event.clear()
                num_rocks += 1
                while len(rocks) < num_rocks:
                    BigRock(screen, rocks)

        if pause:
            pause = fader.info(pause)

        pygame.display.update()


def start_up_with_ai(fader, screen):
    """Run the startup screen with an added AI mode option. Returns True if AI mode selected."""
    global fpsClock

    # Build the info surface with the extra AI line
    screen.fill(WHITE)
    fader.info_blit(False, True)
    textBlit(screen, "or hit <a> for AI mode", "Arial", 40, RED,
             "center", winWidth / 2, 23 * winHeight / 24)
    infoSurf = screen.subsurface(pygame.Rect(0, 0, WIDTH, HEIGHT)).copy()

    ship = Ship(screen)
    bullets = pygame.sprite.Group()
    rocks = pygame.sprite.Group()
    burst = Burst()
    ai_mode = False

    run = True
    starting_up = True

    def rebuild_info_surf():
        screen.fill(WHITE)
        fader.info_blit(False, True)
        textBlit(screen, "or hit <a> for AI mode", "Arial", 40, RED,
                 "center", winWidth / 2, 23 * winHeight / 24)
        return screen.subsurface(pygame.Rect(0, 0, WIDTH, HEIGHT)).copy()

    while run:
        dt = fpsClock.tick(FPS) * REFERENCE_FPS / 1000.0
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN and event.key == K_q:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                if event.key == K_b:
                    burst.shoot = True
                elif event.key == K_p:
                    print('ship is at', ship.p.x, ship.p.y)
            elif event.type == KEYUP:
                if event.key in (K_1, K_2, K_3, K_4):
                    space_map = {K_1: (1, 1), K_2: (1, -1), K_3: (-1, 1), K_4: (-1, -1)}
                    Space.set_space(*space_map[event.key])
                    if event.key != K_1:
                        infoSurf = rebuild_info_surf()
                    del ship
                    ship = Ship(screen)
                elif event.key == K_c:
                    run = False
                elif event.key == K_a:
                    ai_mode = True
                    run = False
                elif event.key == K_SPACE:
                    Bullet(screen, ship, bullets)

        thrust = left = right = False
        keys = pygame.key.get_pressed()
        if keys[K_UP]:
            thrust = True
        if keys[K_LEFT]:
            left = True
        if keys[K_RIGHT]:
            right = True
        if keys[K_DOWN]:
            ship.dx = ship.dy = 0

        screen.fill(WHITE)

        if starting_up and fader.frames > 0:
            fader.title_banner(dt)
        else:
            starting_up = False

        if not starting_up:
            screen.blit(infoSurf, (0, 0, winWidth, winHeight))

        ship.update(thrust, left, right, dt)

        if burst.update(dt):
            Bullet(screen, ship, bullets)

        bullets.update(bullets, rocks, dt)

        pygame.display.update()

    fader.reset()
    bullets.empty()
    rocks.empty()
    del ship

    return ai_mode


if __name__ == '__main__':
    main()
