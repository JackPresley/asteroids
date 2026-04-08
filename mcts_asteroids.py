# mcts_asteroids.py
#
# Monte Carlo Tree Search agent for Asteroids.
#
# Key design choices for real-time play:
#   - Static evaluation at leaf nodes instead of rollouts (fast enough to
#     stay within frame budget so dt stays stable).
#   - Context-based action pruning (branching factor ~4-5 instead of 10).
#   - Time-budgeted search (never exceeds the frame window).
#   - dt capped so physics never blow up even if a search runs long.
#
# No training needed -- plans online using a forward model.
#
# Usage:
#   python mcts_asteroids.py                    # watch MCTS play
#   python mcts_asteroids.py --budget-ms 12     # more search time per move
#   python mcts_asteroids.py --fast             # uncapped FPS
#
import sys
import time
import argparse
import random
from math import sin, cos, pi, sqrt, log, atan2

# ---------------------------------------------------------------------------
# Game constants (must match asteroids.py)
# ---------------------------------------------------------------------------

NUM_ROCKS = 3
WIDTH = 900
HEIGHT = 700
winWidth = WIDTH + 1
winHeight = HEIGHT + 1
FPS = 60
REFERENCE_FPS = 150

WHITE = (255, 255, 255)
BLUE = (100, 149, 237)
RED = (220, 20, 60)

SIM_DT = REFERENCE_FPS / FPS  # ~2.5
MAX_DT = SIM_DT * 1.5         # cap to prevent physics blowup

# ---------------------------------------------------------------------------
# Toroidal helpers
# ---------------------------------------------------------------------------

_DIAG = sqrt((winWidth / 2) ** 2 + (winHeight / 2) ** 2)


def wrap_delta(a, b, mod):
    d = b - a
    if d > mod / 2:
        d -= mod
    elif d < -mod / 2:
        d += mod
    return d


def torus_dist(x1, y1, x2, y2):
    dx = abs(x1 - x2)
    dx = min(dx, winWidth - dx)
    dy = abs(y1 - y2)
    dy = min(dy, winHeight - dy)
    return sqrt(dx * dx + dy * dy)


# ---------------------------------------------------------------------------
# Lightweight sim objects
# ---------------------------------------------------------------------------

SHIP_RADIUS = 18


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


def make_sim_bullet(ship):
    speed = 5
    tdx = -sin(ship.theta * pi / 180)
    tdy = -cos(ship.theta * pi / 180)
    bdx = speed * tdx + ship.d_theta * tdy + ship.dx
    bdy = speed * tdy - ship.d_theta * tdx + ship.dy
    return SimBullet(ship.x, ship.y, bdx, bdy, 6 * winHeight / 7)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

# (thrust, left, right, shoot)
ACTIONS = [
    (False, False, False, False),  # 0  drift
    (True,  False, False, False),  # 1  thrust
    (False, True,  False, False),  # 2  left
    (False, False, True,  False),  # 3  right
    (True,  True,  False, False),  # 4  thrust+left
    (True,  False, True,  False),  # 5  thrust+right
    (False, False, False, True),   # 6  shoot
    (False, True,  False, True),   # 7  left+shoot
    (False, False, True,  True),   # 8  right+shoot
    (True,  False, False, True),   # 9  thrust+shoot
]

N_ACTIONS = len(ACTIONS)
SIM_STEPS_PER_ACTION = 4


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

class GameState:
    __slots__ = ('ship', 'rocks', 'bullets', 'shoot_cooldown',
                 'alive', 'rocks_killed')

    def __init__(self, ship, rocks, bullets, shoot_cooldown=0):
        self.ship = ship
        self.rocks = rocks
        self.bullets = bullets
        self.shoot_cooldown = shoot_cooldown
        self.alive = True
        self.rocks_killed = 0

    def copy(self):
        gs = GameState(
            self.ship.copy(),
            [r.copy() for r in self.rocks],
            [b.copy() for b in self.bullets],
            self.shoot_cooldown,
        )
        gs.rocks_killed = self.rocks_killed
        return gs

    def step(self, action_idx):
        """Advance SIM_STEPS_PER_ACTION physics frames. Returns immediate reward."""
        thrust, left, right, shoot = ACTIONS[action_idx]
        reward = 0.0

        if shoot and self.shoot_cooldown <= 0 and len(self.bullets) < 6:
            self.bullets.append(make_sim_bullet(self.ship))
            self.shoot_cooldown = 15

        for _ in range(SIM_STEPS_PER_ACTION):
            self.ship.step(thrust, left, right)
            for r in self.rocks:
                r.step()

            self.bullets = [b for b in self.bullets if (b.step() or True) and b.dist_left > 0]
            # (step is called via side-effect in the comprehension above --
            #  step() returns None so `or True` keeps the expression truthy)

            if self.shoot_cooldown > 0:
                self.shoot_cooldown -= SIM_DT

            # Bullet-rock collisions
            hit_rocks = set()
            hit_bullets = set()
            for bi, b in enumerate(self.bullets):
                for ri, r in enumerate(self.rocks):
                    if ri not in hit_rocks:
                        if torus_dist(b.x, b.y, r.x, r.y) < r.radius + 5:
                            hit_rocks.add(ri)
                            hit_bullets.add(bi)
                            self.rocks_killed += 1
                            reward += 50.0
                            break

            if hit_rocks:
                new_rocks = []
                for ri in hit_rocks:
                    rock = self.rocks[ri]
                    if rock.radius >= 50:
                        cr, sr = 30, 4
                    elif rock.radius >= 30:
                        cr, sr = 15, 6
                    else:
                        continue
                    for _ in range(2):
                        ddx = ddy = 0
                        while ddx == 0 and ddy == 0:
                            ddx = random.randint(-sr, sr)
                            ddy = random.randint(-sr, sr)
                        new_rocks.append(
                            SimRock(rock.x, rock.y, ddx * 0.2, ddy * 0.2, cr))
                self.rocks = [r for i, r in enumerate(self.rocks)
                              if i not in hit_rocks] + new_rocks
                self.bullets = [b for i, b in enumerate(self.bullets)
                                if i not in hit_bullets]

            # Ship-rock collision
            for r in self.rocks:
                if torus_dist(self.ship.x, self.ship.y, r.x, r.y) < r.radius * 0.7 + SHIP_RADIUS:
                    self.alive = False
                    return reward - 500.0

        return reward


# ---------------------------------------------------------------------------
# Static evaluation (replaces expensive rollouts)
# ---------------------------------------------------------------------------

def static_eval(state):
    """Evaluate a game state cheaply. Higher = better for the agent."""
    if not state.alive:
        return -1000.0

    ship = state.ship
    score = 0.0

    # Credit for kills so far
    score += state.rocks_killed * 50.0

    if not state.rocks:
        return score + 200.0   # wave cleared -- great

    # -- Danger assessment: every rock contributes --
    min_edge_dist = float('inf')
    total_danger = 0.0
    for r in state.rocks:
        edge_dist = torus_dist(ship.x, ship.y, r.x, r.y) - r.radius
        if edge_dist < min_edge_dist:
            min_edge_dist = edge_dist

        if edge_dist < 250:
            dx_to = wrap_delta(ship.x, r.x, winWidth)
            dy_to = wrap_delta(ship.y, r.y, winHeight)
            d = max(1.0, sqrt(dx_to * dx_to + dy_to * dy_to))
            # Closing speed (positive = approaching)
            rel_vx = r.dx - ship.dx
            rel_vy = r.dy - ship.dy
            closing = (dx_to * rel_vx + dy_to * rel_vy) / d
            proximity = max(0.0, 1.0 - edge_dist / 250.0)
            # Scale sharply at close range
            proximity = proximity * proximity
            speed_mult = 1.0 + max(0.0, closing) * 0.8
            # Larger rocks = bigger collision zone = more dangerous
            size_mult = r.radius / 30.0
            total_danger += proximity * speed_mult * size_mult

    # Heavy penalty for danger; this is the primary survival signal
    score -= total_danger * 40.0

    # Bonus for maintaining safe distance
    score += min(min_edge_dist, 300) * 0.15

    # Speed penalty: can't dodge when moving fast
    speed = sqrt(ship.dx ** 2 + ship.dy ** 2)
    if speed > 2.5:
        score -= (speed - 2.5) ** 2 * 2.0

    # Bullets heading toward rocks: reward accurate shots in flight
    for b in state.bullets:
        if b.dist_left <= 0:
            continue
        for r in state.rocks:
            bdx = wrap_delta(b.x, r.x, winWidth)
            bdy = wrap_delta(b.y, r.y, winHeight)
            bd = sqrt(bdx * bdx + bdy * bdy)
            if bd < 1:
                score += 15.0
                continue
            # Dot product of bullet velocity with direction to rock
            dot = (b.dx * bdx + b.dy * bdy) / (b.speed * bd) if b.speed > 0 else 0
            if dot > 0.9 and bd < 300:
                score += 10.0 * dot  # bullet is on target

    return score


# ---------------------------------------------------------------------------
# Action pruning
# ---------------------------------------------------------------------------

def prune_actions(state):
    """Return a list of action indices worth considering in this state."""
    can_shoot = state.shoot_cooldown <= 0 and len(state.bullets) < 6
    ship = state.ship

    if not state.rocks:
        # No rocks: drift, or brake if moving fast
        speed = sqrt(ship.dx ** 2 + ship.dy ** 2)
        if speed > 1.5:
            return [0, 1, 2, 3, 4, 5]  # all movement actions to find braking
        return [0]  # just drift

    # Always consider: drift, thrust, turn-left, turn-right, thrust+left, thrust+right
    actions = [0, 1, 2, 3, 4, 5]

    if can_shoot:
        # Add shoot combos
        actions.extend([6, 7, 8, 9])

    return actions


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

UCB_C = 1.0   # exploration constant, tuned down for tactical game


class MCTSNode:
    __slots__ = ('state', 'action', 'parent', 'children',
                 'visits', 'total_value', 'untried')

    def __init__(self, state, action=None, parent=None, available_actions=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_value = 0.0
        if available_actions is not None:
            self.untried = list(available_actions)
        else:
            self.untried = prune_actions(state)
        random.shuffle(self.untried)

    def is_fully_expanded(self):
        return len(self.untried) == 0

    def is_terminal(self):
        return not self.state.alive

    def ucb1(self):
        if self.visits == 0:
            return float('inf')
        return (self.total_value / self.visits
                + UCB_C * sqrt(log(self.parent.visits) / self.visits))

    def best_child_ucb(self):
        return max(self.children, key=lambda c: c.ucb1())

    def best_child_visits(self):
        return max(self.children, key=lambda c: c.visits)

    def expand(self):
        action = self.untried.pop()
        child_state = self.state.copy()
        child_state.step(action)
        child = MCTSNode(child_state, action=action, parent=self)
        self.children.append(child)
        return child

    def backpropagate(self, value):
        node = self
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent


def mcts_search(root_state, budget_sec=0.008):
    """Time-budgeted MCTS. Returns best action index."""
    root = MCTSNode(root_state.copy())
    deadline = time.monotonic() + budget_sec
    iterations = 0

    while time.monotonic() < deadline:
        node = root

        # Selection
        while node.is_fully_expanded() and node.children and not node.is_terminal():
            node = node.best_child_ucb()

        # Expansion
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()

        # Evaluation (static -- no rollout)
        value = static_eval(node.state)

        # Backpropagation
        node.backpropagate(value)
        iterations += 1

    if not root.children:
        # Fallback: heuristic
        return _heuristic_action(root_state)

    return root.best_child_visits().action


# ---------------------------------------------------------------------------
# Fallback heuristic (only used if MCTS has zero iterations somehow)
# ---------------------------------------------------------------------------

def _heuristic_action(state):
    ship = state.ship
    if not state.rocks:
        return 0

    nearest = min(state.rocks,
                  key=lambda r: torus_dist(ship.x, ship.y, r.x, r.y))
    dist = torus_dist(ship.x, ship.y, nearest.x, nearest.y)

    if dist < 100:
        dx_to = wrap_delta(ship.x, nearest.x, winWidth)
        dy_to = wrap_delta(ship.y, nearest.y, winHeight)
        escape = atan2(dx_to, dy_to) * 180 / pi   # away from rock
        diff = (escape - ship.theta + 180) % 360 - 180
        if abs(diff) < 45:
            return 1
        return 4 if diff > 0 else 5

    # Aim at nearest
    dx_to = wrap_delta(ship.x, nearest.x, winWidth)
    dy_to = wrap_delta(ship.y, nearest.y, winHeight)
    aim = atan2(-dx_to, -dy_to) * 180 / pi
    diff = (aim - ship.theta + 180) % 360 - 180
    if abs(diff) < 10 and state.shoot_cooldown <= 0:
        return 6
    return 2 if diff > 0 else 3


# ---------------------------------------------------------------------------
# Build sim state from live pygame sprites
# ---------------------------------------------------------------------------

def build_sim_state(ship, rocks, bullets):
    import asteroids as _astro
    s = SimShip(ship.p.x, ship.p.y, ship.dx, ship.dy, ship._theta)
    s.d_theta = ship.d_theta
    rs = []
    for r in rocks:
        radius = 50
        if isinstance(r, _astro.SmallRock):
            radius = 15
        elif isinstance(r, _astro.MediumRock):
            radius = 30
        rs.append(SimRock(r.p.x, r.p.y, r.dx, r.dy, radius))
    bs = []
    for b in bullets:
        remaining = b.distance - b.distance_travelled
        bs.append(SimBullet(b.p.x, b.p.y, b.dx, b.dy, remaining))
    return s, rs, bs


# ---------------------------------------------------------------------------
# Watch mode
# ---------------------------------------------------------------------------

def watch(budget_ms=8, fast=False):
    import pygame
    from pygame.locals import QUIT, KEYUP, K_q
    import asteroids as _astro

    pygame.init()
    fpsClock = pygame.time.Clock()
    _astro.fpsClock = fpsClock
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Asteroids - MCTS Agent")
    pygame.mouse.set_visible(0)

    Ship     = _astro.Ship
    Bullet   = _astro.Bullet
    BigRock  = _astro.BigRock
    Score    = _astro.Score
    textBlit = _astro.textBlit

    ship = Ship(screen)
    bullets = pygame.sprite.Group()
    rocks   = pygame.sprite.Group()
    num_rocks = NUM_ROCKS
    Score.reset()

    while len(rocks) < num_rocks:
        BigRock(screen, rocks)

    shoot_cooldown = 0
    frame_count = 0
    current_action = ACTIONS[0]
    budget_sec = budget_ms / 1000.0

    while True:
        raw_ms = fpsClock.tick(0 if fast else FPS)
        # CAP dt so physics stays stable even if a frame ran long
        dt = min(raw_ms * REFERENCE_FPS / 1000.0, MAX_DT)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYUP and event.key == K_q:
                pygame.quit()
                sys.exit()

        # MCTS decision every SIM_STEPS_PER_ACTION frames
        if frame_count % SIM_STEPS_PER_ACTION == 0:
            s, rs, bs = build_sim_state(ship, rocks, bullets)
            game_state = GameState(s, rs, bs, shoot_cooldown)
            action_idx = mcts_search(game_state, budget_sec=budget_sec)
            current_action = ACTIONS[action_idx]

        thrust, left, right, shoot = current_action

        if shoot_cooldown > 0:
            shoot_cooldown -= dt
        if shoot and shoot_cooldown <= 0 and len(bullets) < 6:
            fire = True
            shoot_cooldown = 15
        else:
            fire = False

        screen.fill(WHITE)
        ship.update(thrust, left, right, dt)

        if fire:
            Bullet(screen, ship, bullets)

        if bullets:
            bullets.update(bullets, rocks, dt)
        rocks.update(dt)

        if pygame.sprite.spritecollideany(
                ship, rocks, pygame.sprite.collide_circle_ratio(0.7)):
            Score.delLife()
            if Score.getLives() == 0:
                textBlit(screen, f"Final score: {Score.get()}", "Arial", 60,
                         RED, "center", winWidth / 2, winHeight / 2)
                pygame.display.update()
                pygame.time.wait(3000)
                Score.reset()
                num_rocks = NUM_ROCKS
            bullets.empty()
            rocks.empty()
            del ship
            pygame.event.clear()
            ship = Ship(screen)
            while len(rocks) < num_rocks:
                BigRock(screen, rocks)
            shoot_cooldown = 0

        Score.draw(screen, rocks)

        # Show iteration count from last search
        textBlit(screen, "MCTS Agent", "Arial", 30, BLUE,
                 "bottomleft", winWidth / 20, 18 * winHeight / 20, False)

        if len(rocks) == 0:
            num_rocks += 1
            bullets.empty()
            pygame.event.clear()
            while len(rocks) < num_rocks:
                BigRock(screen, rocks)

        pygame.display.update()
        frame_count += 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCTS agent for Asteroids")
    parser.add_argument("--budget-ms", type=int, default=8,
                        help="Time budget per MCTS decision in ms (default: 8)")
    parser.add_argument("--fast", action="store_true",
                        help="Run without FPS cap")
    args = parser.parse_args()

    watch(budget_ms=args.budget_ms, fast=args.fast)
