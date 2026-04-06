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
SIM_STEPS_PER_ACTION = 10  # frames per action choice
DEPTH = 2


def evaluate(ship, rocks, bullets):
    """Heuristic evaluation of a game state. Higher is better."""
    if not rocks:
        return 10000

    score = 0.0

    # Survival: penalise proximity to rocks, amplified by closing speed.
    # A rock approaching from the side or rear is just as dangerous as one
    # head-on — the closing speed term captures this regardless of direction.
    min_dist = float('inf')
    for r in rocks:
        rdx = wrap_delta(ship.x, r.x, winWidth)
        rdy = wrap_delta(ship.y, r.y, winHeight)
        dist_full = sqrt(rdx * rdx + rdy * rdy)
        d = dist_full - r.radius - SHIP_RADIUS
        if d < min_dist:
            min_dist = d
        if d < 0:
            return -100000  # collision — terrible

        # Closing speed: rate at which rock and ship are approaching each other.
        # Positive = closing, negative = separating.
        if dist_full > 1e-6:
            ux, uy = rdx / dist_full, rdy / dist_full
            closing = -((r.dx - ship.dx) * ux + (r.dy - ship.dy) * uy)
        else:
            closing = 0.0

        # Scale penalty upward for approaching rocks; separating rocks are less urgent.
        closing_factor = max(1.0, 1.0 + closing * 4.0)
        if d < 40:
            score -= 5000 / (d + 1) * closing_factor
        elif d < 120:
            score -= 500 / (d + 1) * closing_factor

        # Time-to-impact warning: penalise rocks that will reach us within ~50 frames
        # even if they are currently outside the immediate danger zone.
        if closing > 0.05:
            tti = d / closing
            if tti < 50:
                score -= 4000 / (tti + 1)

    # Reward keeping distance from rocks
    score += min(min_dist, 300)

    # Multi-rock convergence penalty: when N rocks are simultaneously closing in
    # the danger is superlinear because no single escape vector works for all of them.
    converging = 0
    for r in rocks:
        rdx = wrap_delta(ship.x, r.x, winWidth)
        rdy = wrap_delta(ship.y, r.y, winHeight)
        dist_full = sqrt(rdx * rdx + rdy * rdy)
        gap = dist_full - r.radius - SHIP_RADIUS
        if gap < 150 and dist_full > 1e-6:
            ux, uy = rdx / dist_full, rdy / dist_full
            closing = -((r.dx - ship.dx) * ux + (r.dy - ship.dy) * uy)
            if closing > 0.05:
                converging += 1
    if converging >= 2:
        score -= 600 * (converging - 1) ** 2

    # Reward bullets closing on rocks (closing-speed aware)
    for b in bullets:
        for r in rocks:
            bd = torus_dist(b.x, b.y, r.x, r.y)
            if bd < r.radius + 5:
                score += 600  # about to hit
            elif bd < r.radius + 60:
                score += 200
            else:
                rdx = wrap_delta(b.x, r.x, winWidth)
                rdy = wrap_delta(b.y, r.y, winHeight)
                cs = -(b.dx * rdx + b.dy * rdy) / bd if bd > 1e-6 else 0
                if cs > 2.0:
                    score += 120

    # Penalise high speed (quadratic above 0.8, matching neat_asteroids tuning)
    speed = sqrt(ship.dx ** 2 + ship.dy ** 2)
    if speed > 0.8:
        _excess = speed - 0.8
        score -= 0.60 * _excess * _excess * 200  # scaled to ~match linear at speed 2

    # Reward aiming at the intercept point of the best untargeted rock.
    # Skip rocks that already have a bullet closing on them so the ship
    # doesn't waste turns re-aiming at a rock that is about to be destroyed.
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
    aim_diff = abs(intercept_aim_angle(ship, nearest))
    if aim_diff < 5:
        score += 150
    elif aim_diff < 15:
        score += 80
    elif aim_diff < 30:
        score += 30

    # Reward retrograde thrust (braking): ship heading opposite to velocity
    if speed > 0.1:
        tdx = -sin(ship.theta * pi / 180)
        tdy = -cos(ship.theta * pi / 180)
        retro_dot = -(tdx * (ship.dx / speed) + tdy * (ship.dy / speed))
        if retro_dot > 0:
            score += 300 * retro_dot * min(1.0, speed / 3.0)

    # Global open-space 4×3 grid search: reward being near the safest cell
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

    # Velocity toward safe zone: reward heading toward the safest cell when not already there.
    # This encourages proactive repositioning rather than only rewarding arrival.
    dist_to_safe = torus_dist(ship.x, ship.y, best_x, best_y)
    if dist_to_safe > 80 and speed > 0.05:
        sdx = wrap_delta(ship.x, best_x, winWidth)
        sdy = wrap_delta(ship.y, best_y, winHeight)
        safe_ux = sdx / dist_to_safe
        safe_uy = sdy / dist_to_safe
        vel_toward = (ship.dx * safe_ux + ship.dy * safe_uy) / speed
        if vel_toward > 0:
            score += 150 * vel_toward * min(1.0, dist_to_safe / 200)

    # Centre return: graded by rock count; last-rock sequence adds big rewards
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


def _order_actions(ship, rocks, bullets):
    """Return action indices sorted by quick 1-step evaluation (best first).

    This lets alpha-beta pruning cut more branches at deeper levels.
    """
    scored = []
    for ai, action in enumerate(ACTIONS):
        new_ship, new_rocks, new_bullets, dead = sim_step(ship, rocks, bullets, action)
        s = -100000 if dead else evaluate(new_ship, new_rocks, new_bullets)
        scored.append((s, ai))
    scored.sort(reverse=True)
    return [ai for _, ai in scored]


def minimax(ship, rocks, bullets, depth, bound=float('inf')):
    """Return (best_score, best_action_index).

    bound: upper bound from the caller — if we find a score >= bound we can
    stop early because the caller (a max node) already has a branch that good
    and will never choose this one.  Equivalent to beta in alpha-beta pruning
    for a single-player max tree.
    """
    if depth == 0 or not rocks:
        return evaluate(ship, rocks, bullets), 0

    # Apply move ordering only at the root to maximise pruning there without
    # doubling work at every internal node.
    if depth == DEPTH:
        ordered = _order_actions(ship, rocks, bullets)
    else:
        ordered = range(len(ACTIONS))

    best_score = float('-inf')
    best_action = 0

    for ai in ordered:
        action = ACTIONS[ai]
        new_ship, new_rocks, new_bullets, dead = sim_step(ship, rocks, bullets, action)
        if dead:
            action_score = -100000
        elif depth == 1:
            action_score = evaluate(new_ship, new_rocks, new_bullets)
        else:
            action_score, _ = minimax(new_ship, new_rocks, new_bullets, depth - 1, best_score)

        if action_score > best_score:
            best_score = action_score
            best_action = ai
            if best_score >= bound:
                break  # parent already has a better option; prune remaining actions

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
