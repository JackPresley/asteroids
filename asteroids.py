# asteroids.py                                                        Scott Simmons Spring 2015
#
# Copyright 2015 Scott Simmons
#
import pygame, sys
from pygame.locals import *
from random import randint
from math import sin, cos, pi

"""
KNOWN ISSUES:

  - collisions aren't exactly right on the boundary but that might not matter.

  - the rocks may only start out traveling in a small number of directions.  Does that matter?

  - check to see if objects (like bullets) are being deleted correctly.

  - did not finish making the rocks look good.

"""

NUM_ROCKS = 3  # number of rocks at beginning of game

WIDTH = 900
HEIGHT = 700

winWidth = WIDTH + 1
winHeight = HEIGHT + 1

FPS = 60
REFERENCE_FPS = 150  # speeds were tuned at this rate; dt is normalized to it

WHITE = (255, 255, 255)
GREY = (80, 80, 80)
BLACK = (0, 0, 0)
BLUE = (100, 149, 237)
RED = (220, 20, 60)


class Space:
    """
    Create the correct topological space.

    t_x = -1 half twists while gluing along the x-axis; likewise for t_y
    """

    t_x = 1  # class variables
    t_y = 1

    def __init__(self, x, y):
        self.flipped_x = 0
        self.flipped_y = 0
        self.set_coords(x, y)

    # Note: instance attributes relf_x and relf_y get switched back to false by MySprite
    #       should they become True
    def set_coords(self, x, y):
        self.refl_x = (x < 0 or x >= winWidth) and Space.t_y == -1
        self.refl_y = (y < 0 or y >= winHeight) and Space.t_x == -1
        if self.refl_x:
            self.y = (Space.t_y * y) % winHeight
        else:
            self.y = y % winHeight
        if self.refl_y:
            self.x = (Space.t_x * x) % winWidth
        else:
            self.x = x % winWidth
        if self.refl_x:
            self.flipped_x = (self.flipped_x + 1) % 2
        if self.refl_y:
            self.flipped_y = (self.flipped_y + 1) % 2
        self.flipped = (self.flipped_x + self.flipped_y) % 2

    @classmethod
    def set_space(cls, t_x, t_y):
        cls.t_x = t_x
        cls.t_y = t_y


class MySprite(pygame.sprite.Sprite):
    """
    This class adds some functionality to Pygame's built-in Sprite class.  Mainly, MySprite
    instances can draw themselves to a surface such as a torus (or a Klein bottle, or
    projective plane, as the case may be, and can be animated.

    - images must be a dictionary such as

         { 0 : 'image.png', 1 : 'image.png' } or  { 'left': 'shipleft.png', etc }

      or { 0 : 'image.png' } for a static sprite

    - an instance of MySprite moves, each game loop, the length and direction of the vector
      <dx, dy> and is oriented (rotated) according to theta.  It is drawn with center at (x,y).

    - it's best if all the images are the same size

    - when extending this class, make sure to call update_position() before draw()
    """

    def __init__(self, screen, images, x, y, theta, dx, dy, d_theta):
        super().__init__()
        self.screen = screen
        self._images = images
        self.p = Space(x, y)
        self._theta = theta
        self.dx, self.dy = dx, dy
        self.d_theta = d_theta  # change in rotation angle per game loop
        self.rect = self._images[list(self._images.keys()).pop()].get_rect()

    def draw(self, key):
        rotImage = pygame.transform.rotate(
            self._images[key], self._theta
        )  # rotate the image
        rotRect = rotImage.get_rect()
        rotRectCenter = (
            rotRect.center
        )  # the center of the bounding box for the rotated image
        rotRectWidth = (
            rotRect.width
        )  #   Unless the rotation angle is quadrantal, this bounding
        rotRectHeight = (
            rotRect.height
        )  #   box will be larger that the original bounding box

        rot_x = self.p.x - rotRectCenter[0]  # With these upper_left coordinates
        rot_y = self.p.y - rotRectCenter[1]  # the center of the image is preserved

        self.screen.blit(rotImage, (rot_x, rot_y))

        temp = rotImage
        if self.p.x > winWidth - rotRectCenter[0]:  # the right edge
            pp = Space(winWidth, self.p.y)
            if pp.refl_x:
                rotImage = pygame.transform.flip(rotImage, False, True)
            self.screen.blit(
                rotImage,
                (0, pp.y - rotRectCenter[1]),
                (winWidth - rot_x, 0, rotRectWidth - (winWidth - rot_x), rotRectHeight),
            )
        elif self.p.x < rotRectCenter[0]:  # the left edge
            pp = Space(rot_x, self.p.y)
            if pp.refl_x:
                rotImage = pygame.transform.flip(rotImage, False, True)
            self.screen.blit(
                rotImage,
                (winWidth + rot_x, pp.y - rotRectCenter[1]),
                (0, 0, -rot_x, rotRectHeight),
            )
        rotImage = temp
        if self.p.y > winHeight - rotRectCenter[1]:  # the bottom edge
            pp = Space(self.p.x, winHeight)
            if pp.refl_y:
                rotImage = pygame.transform.flip(rotImage, True, False)
            self.screen.blit(
                rotImage,
                (pp.x - rotRectCenter[0], 0),
                (
                    0,
                    winHeight - rot_y,
                    rotRectWidth,
                    rotRectHeight - (winHeight - rot_y),
                ),
            )
        elif self.p.y < rotRectCenter[1]:  # the top edge
            pp = Space(self.p.x, rot_y)
            if pp.refl_y:
                rotImage = pygame.transform.flip(rotImage, True, False)
            self.screen.blit(
                rotImage,
                (pp.x - rotRectCenter[0], winHeight + rot_y),
                (0, 0, rotRectHeight, -rot_y),
            )
        rotImage = temp

        # Now the corners
        if self.p.x < rotRectCenter[0] and self.p.y < rotRectCenter[1]:  # top left
            pp = Space(rot_x, rot_y)
            if Space.t_x == Space.t_y == 1:  # torus
                self.screen.blit(rotImage, (pp.x, pp.y), (0, 0, -rot_x, -rot_y))
            if Space.t_x == 1 and Space.t_y == -1:  # 1, -1 Klein bottle
                flipImage = pygame.transform.flip(rotImage, False, True)
                self.screen.blit(
                    flipImage, (pp.x, 0), (0, rotRectHeight + rot_y, -rot_x, -rot_y)
                )
            if Space.t_x == -1 and Space.t_y == 1:  # -1, 1 Klein bottle
                flipImage = pygame.transform.flip(rotImage, True, False)
                self.screen.blit(
                    flipImage, (0, pp.y), (rotRectWidth + rot_x, 0, -rot_x, -rot_y)
                )

        elif (
            self.p.x < rotRectCenter[0] and self.p.y > winHeight - rotRectCenter[1]
        ):  # bottom left
            pp = Space(rot_x, rot_y + rotRectHeight)
            if Space.t_x == Space.t_y == 1:  # torus
                self.screen.blit(
                    rotImage,
                    (pp.x, 0),
                    (0, winHeight - rot_y, -rot_x, rot_y + rotRectHeight - winHeight),
                )
            if Space.t_x == 1 and Space.t_y == -1:  # 1, -1 Klein bottle
                flipImage = pygame.transform.flip(rotImage, False, True)
                self.screen.blit(flipImage, (pp.x, pp.y), (0, 0, -rot_x, -rot_y))
            if Space.t_x == -1 and Space.t_y == 1:  # -1, 1 Klein bottle
                flipImage = pygame.transform.flip(rotImage, True, False)
                self.screen.blit(
                    flipImage,
                    (0, 0),
                    (
                        rotRectWidth + rot_x,
                        winHeight - rot_y,
                        -rot_x,
                        rot_y + rotRectHeight - winHeight,
                    ),
                )

        elif (
            self.p.x > winWidth - rotRectCenter[0]
            and self.p.y > winHeight - rotRectCenter[1]
        ):  # bottom right
            pp = Space(rot_x + rotRectWidth, rot_y + rotRectHeight)
            if Space.t_x == Space.t_y == 1:  # torus
                self.screen.blit(
                    rotImage,
                    (0, 0),
                    (
                        winWidth - rot_x,
                        winHeight - rot_y,
                        rot_x + rotRectWidth - winWidth,
                        rot_y + rotRectHeight - winHeight,
                    ),
                )
            if Space.t_x == 1 and Space.t_y == -1:  # 1, -1 Klein bottle
                flipImage = pygame.transform.flip(rotImage, False, True)
                self.screen.blit(
                    flipImage,
                    (0, pp.y),
                    (
                        winWidth - rot_x,
                        0,
                        rot_x + rotRectWidth - winWidth,
                        rot_y + rotRectHeight - winHeight,
                    ),
                )
            if Space.t_x == -1 and Space.t_y == 1:  # -1, 1 Klein bottle
                flipImage = pygame.transform.flip(rotImage, True, False)
                self.screen.blit(
                    flipImage,
                    (pp.x, 0),
                    (
                        0,
                        winHeight - rot_y,
                        rot_x + rotRectWidth - winWidth,
                        rot_y + rotRectHeight - winHeight,
                    ),
                )

        elif (
            self.p.x > winWidth - rotRectCenter[0] and self.p.y < rotRectCenter[1]
        ):  # top right
            pp = Space(rot_x + rotRectWidth, rot_y)
            if Space.t_x == Space.t_y == 1:  # torus
                self.screen.blit(
                    rotImage,
                    (0, pp.y),
                    (winWidth - rot_x, 0, rot_x + rotRectWidth - winWidth, -rot_y),
                )
            if Space.t_x == 1 and Space.t_y == -1:  # 1, -1 Klein bottle
                flipImage = pygame.transform.flip(rotImage, False, True)
                self.screen.blit(
                    flipImage,
                    (0, 0),
                    (
                        winWidth - rot_x,
                        rotRectHeight + rot_y,
                        rot_x + rotRectWidth - winWidth,
                        -rot_y,
                    ),
                )
            if Space.t_x == -1 and Space.t_y == 1:  # -1, 1 Klein bottle
                flipImage = pygame.transform.flip(rotImage, True, False)
                self.screen.blit(
                    flipImage,
                    (pp.x, pp.y),
                    (0, 0, rot_x + rotRectWidth - winWidth, -rot_y),
                )

    def update_position(self, dt=1):
        self.p.set_coords(self.p.x + self.dx * dt, self.p.y + self.dy * dt)
        if self.p.refl_x:
            self.dy = -self.dy
            self._theta = 180 - self._theta
            self.d_theta = -self.d_theta
            self.p.refl_x = False
        if self.p.refl_y:
            self.dx = -self.dx
            self._theta = -self._theta
            self.d_theta = -self.d_theta
            self.p.refl_y = False
        self._theta += self.d_theta * dt
        self.rect.center = self.p.x, self.p.y


class Ship(MySprite):  # This class extends the MySprite class defined above
    _images_cache = None

    @classmethod
    def _init_images(cls):
        if cls._images_cache is None:
            cls._images_cache = {}
            cls._images_cache["off"] = pygame.image.load("spaceShip2.png").convert_alpha()
            cls._images_cache["right"] = pygame.image.load("spaceShip2right.png").convert_alpha()
            cls._images_cache["left"] = pygame.transform.flip(cls._images_cache["right"], True, False)
            cls._images_cache["both"] = pygame.image.load("spaceShip2thrust.png").convert_alpha()

    def __init__(
        self,
        screen,
        x=winWidth / 2,
        y=winHeight / 2,
        theta=0,
        dx=0,
        dy=0,
        d_theta=0,
        accel=0.02,
    ):
        self.accel = accel
        Ship._init_images()
        super().__init__(screen, Ship._images_cache, x, y, theta, dx, dy, d_theta)

    def update(self, thrust, left, right, dt=1):
        self.d_theta = 0
        self._theta_dx = -sin(self._theta * pi / 180)
        self._theta_dy = -cos(self._theta * pi / 180)
        engines = "off"
        if thrust:
            engines = "both"
            self.dx += self.accel * dt * self._theta_dx
            self.dy += self.accel * dt * self._theta_dy
        if right:
            self.d_theta = -1.5
            engines = "left" if self.p.flipped else "right"
        elif left:
            self.d_theta = 1.5
            engines = "right" if self.p.flipped else "left"
        if self.p.flipped:
            self.d_theta = -self.d_theta

        self.update_position(dt)
        self.draw(engines)


class Bullet(MySprite):
    _image_cache = None
    RADIUS = 5

    @classmethod
    def _init_image(cls):
        if cls._image_cache is None:
            r = cls.RADIUS
            image = pygame.Surface(
                [2 * r, 2 * r], pygame.SRCALPHA, 32
            ).convert_alpha()
            image.fill((255, 255, 255, 0))
            pygame.draw.circle(image, (0, 0, 0), (r, r), r)
            cls._image_cache = image

    def __init__(self, screen, ship, bullets, speed=5, distance=6 * winHeight / 7):
        self.speed = speed  # distance travelled by the bullet per game loop
        self.distance = distance  # distance the bullet travels before disappearing
        self.min_dist = ship.rect.width / 2
        #  x = ship.p.x-ship.rect[2]/2*sin(ship._theta*pi/180)
        #  y = ship.p.y-ship.rect[3]/2*cos(ship._theta*pi/180)
        x = ship.p.x
        y = ship.p.y
        dx = speed * ship._theta_dx + ship.d_theta * ship._theta_dy + ship.dx
        dy = speed * ship._theta_dy - ship.d_theta * ship._theta_dx + ship.dy
        theta = ship._theta
        Bullet._init_image()
        image = Bullet._image_cache
        self.rect = image.get_rect()
        super().__init__(screen, {0: image}, x, y, theta, dx, dy, 0)
        bullets.add(self)
        self.distance_travelled = 0

    def update(self, bullets, rocks, dt=1):
        rock = pygame.sprite.spritecollideany(self, rocks, pygame.sprite.collide_circle)
        if rock != None:
            bullets.remove(self)
            rocks.remove(rock)
            rock.destroy(rock.p.x, rock.p.y)
        else:
            self.distance_travelled += self.speed * dt
            if self.distance_travelled <= self.distance:
                self.update_position(dt)
                if self.distance_travelled >= self.min_dist:
                    self.draw(0)
            else:
                bullets.remove(self)


class Rock(MySprite):
    def __init__(self, screen, images, group, radius, x, y, dx, dy, d_theta):
        self.group = group
        theta = randint(0, 359)
        #   d_theta = .5*d_theta
        super().__init__(screen, images, x, y, theta, dx, dy, d_theta)
        group.add(self)

    def update(self, dt=1):
        self.update_position(dt)
        self.draw(0)


class BigRock(Rock):
    _base_image = None

    @classmethod
    def _init_image(cls, radius=50):
        if cls._base_image is None:
            image = pygame.Surface(
                [2 * radius, 2 * radius], pygame.SRCALPHA, 32
            ).convert_alpha()
            image.fill((255, 255, 255, 0))
            pygame.draw.polygon(
                image,
                GREY,
                [
                    [91, 80],
                    [81, 75],
                    [80, 96],
                    [35, 100],
                    [40, 92],
                    [5, 81],
                    [10, 60],
                    [8, 45],
                    [22, 22],
                    [25, 34],
                    [38, 33],
                    [27, 15],
                    [65, 10],
                    [95, 50],
                    [100, 60],
                ],
                5,
            )
            cls._base_image = image

    def __init__(self, screen, group, slow=0.2, radius=50):
        if randint(0, 1):
            x = randint(-int(winWidth / 20), int(winWidth / 20))
            y = randint(0, winHeight)
        else:
            x = randint(0, winWidth)
            y = randint(-int(winWidth / 20), int(winHeight / 20))
        d_theta = randint(-2, 2)
        dx = dy = 0
        while dx == 0 and dy == 0:
            dx = randint(-3, 3)
            dy = randint(-3, 3)
        dx = slow * dx
        dy = slow * dy

        BigRock._init_image(radius)
        image = BigRock._base_image.copy()
        images = {}
        if randint(0, 1):
            image = pygame.transform.flip(image, True, False)
        images[0] = image
        images[1] = pygame.transform.flip(image, True, False)
        super().__init__(screen, images, group, radius, x, y, dx, dy, d_theta)

    def update(self, dt=1):
        self.update_position(dt)
        if self.p.flipped:
            self.draw(1)
        else:
            self.draw(0)

    def destroy(self, x, y):
        Score.add(5)
        MediumRock(self.screen, self.group, x, y)
        MediumRock(self.screen, self.group, x, y)


class MediumRock(Rock):
    _base_image = None

    @classmethod
    def _init_image(cls, radius=30):
        if cls._base_image is None:
            image = pygame.Surface(
                [2 * radius, 2 * radius], pygame.SRCALPHA, 32
            ).convert_alpha()
            image.fill((255, 255, 255, 0))
            pygame.draw.polygon(
                image, GREY, [[60, 0], [60, 60], [0, 60], [0, 0], [60, 0]], 8
            )
            cls._base_image = image

    def __init__(self, screen, group, x, y, slow=0.2, radius=30):
        d_theta = randint(-3, 3)
        dx = dy = 0
        while dx == 0 and dy == 0:
            dx = randint(-4, 4)
            dy = randint(-4, 4)
        dx, dy = slow * dx, slow * dy

        MediumRock._init_image(radius)
        image = MediumRock._base_image.copy()
        if randint(0, 1):
            image = pygame.transform.flip(image, True, False)
        super().__init__(screen, {0: image}, group, radius, x, y, dx, dy, d_theta)

    def destroy(self, x, y):
        Score.add(10)
        SmallRock(self.screen, self.group, x, y)
        SmallRock(self.screen, self.group, x, y)


class SmallRock(Rock):
    _base_image = None

    @classmethod
    def _init_image(cls, radius=15):
        if cls._base_image is None:
            image = pygame.Surface(
                [2 * radius, 2 * radius], pygame.SRCALPHA, 32
            ).convert_alpha()
            image.fill((255, 255, 255, 0))
            pygame.draw.polygon(
                image, GREY, [[30, 0], [30, 30], [0, 30], [0, 0], [30, 0]], 8
            )
            cls._base_image = image

    def __init__(self, screen, group, x, y, slow=0.2, radius=15):
        d_theta = randint(-4, 4)
        dx = dy = 0
        while dx == 0 and dy == 0:
            dx = randint(-6, 6)
            dy = randint(-6, 6)
        dx, dy = slow * dx, slow * dy

        SmallRock._init_image(radius)
        image = SmallRock._base_image.copy()
        if randint(0, 1):
            image = pygame.transform.flip(image, True, False)
        super().__init__(screen, {0: image}, group, radius, x, y, dx, dy, d_theta)

    def destroy(self, x, y):
        Score.add(20)


_font_cache = {}


def textBlit(
    screen, string, font, font_size, color, loc_string, loc_x, loc_y, antialias=True
):
    key = (font, font_size)
    if key not in _font_cache:
        _font_cache[key] = pygame.font.SysFont(font, font_size)
    _font = _font_cache[key]
    surf = _font.render(string, antialias, color)
    surfRect = surf.get_rect()
    if loc_string == "center":
        surfRect.center = loc_x, loc_y
    elif loc_string == "topleft":
        surfRect.topleft = loc_x, loc_y
    elif loc_string == "bottomleft":
        surfRect.bottomleft = loc_x, loc_y
    screen.blit(surf, surfRect)


class Score:
    score = 0
    lives = 1

    @classmethod
    def draw(cls, screen, rocks):
        textBlit(
            screen,
            "Lives: " + str(cls.lives),
            "Arial",
            35,
            RED,
            "topleft",
            winWidth / 20,
            winHeight / 20,
            False,
        )
        textBlit(
            screen,
            "Score: " + str(cls.score),
            "Arial",
            35,
            RED,
            "topleft",
            4 * winWidth / 20,
            winHeight / 20,
            False,
        )
        textBlit(
            screen,
            "Rocks: " + str(len(rocks)),
            "Arial",
            35,
            RED,
            "bottomleft",
            winWidth / 20,
            19 * winHeight / 20,
            False,
        )

    @classmethod
    def add(cls, amount):
        cls.score += amount

    @classmethod
    def reset(cls):
        cls.score = 0
        cls.lives = 1

    @classmethod
    def get(cls):
        return cls.score

    @classmethod
    def getLives(cls):
        return cls.lives

    @classmethod
    def addLife(cls):
        cls.lives += 1

    @classmethod
    def delLife(cls):
        cls.lives -= 1


# use Fader for blitting fading/shrinking and other stuff onto the screen
class Fader:
    def __init__(self, screen):
        self.screen = screen
        self.frames_max = 420
        assert (
            self.frames_max <= 510
        ), "self.frames_max should probably be less than 510"
        self.frames = self.frames_max  # frames left counting down from self.frames_max
        self.max_font_size = 60  # max font size
        self.font_size = self.max_font_size
        self._bonus_font = None  # lazily created in lifeBonus
        self._life_given = False

    # does not stop game play
    def lifeBonus(self, dt=1):
        # Shrinks to nothing
        # font = pygame.font.Font(None, self.font_size)
        # surf = font.render("BONUS LIFE!",True,RED)

        if self._bonus_font is None:
            self._bonus_font = pygame.font.SysFont("Arial", self.max_font_size)
        font = self._bonus_font

        # Fades from red to white
        color = (
            0 if self.frames > 255 else 255 - self.frames
        )  # color stays at zero for while
        if color < 220:  # and then counts up to 255
            surf = font.render("BONUS LIFE!", True, (220, int(color), int(color)))
        else:
            surf = font.render("BONUS LIFE!", True, (int(color), int(color), int(color)))

        # Fades from blue to white
        color = 100 if self.frames > 255 else 255 - self.frames
        if color < 100:
            surf = font.render("BONUS LIFE!", True, (100, 149, 237))
        elif color < 149:
            surf = font.render("BONUS LIFE!", True, (int(color), 149, 237))
        elif color < 237:
            surf = font.render("BONUS LIFE!", True, (int(color), int(color), 237))
        else:
            surf = font.render("BONUS LIFE!", True, (int(color), int(color), int(color)))

        surfRect = surf.get_rect()
        surfRect.center = (winWidth / 2, winHeight / 3)
        self.screen.blit(surf, surfRect)
        if self.frames <= 2 * self.max_font_size:
            self.font_size -= 0.5 * dt
        self.frames -= dt
        if not self._life_given and self.frames <= 300:
            self._life_given = True
            Score.addLife()

    # stops game play
    def shipDestroyed(self, frozenSurf, dt=1):
        size = max(1, int(self.font_size))
        key = ("Arial", size)
        if key not in _font_cache:
            _font_cache[key] = pygame.font.SysFont("Arial", size)
        font = _font_cache[key]
        surf = font.render("SHIP DESTROYED!", True, BLUE)
        surfRect = surf.get_rect()
        surfRect.center = (winWidth / 2, 0.3 * winHeight)
        self.screen.blit(frozenSurf, (0, 0, winWidth, winHeight))
        self.screen.blit(surf, surfRect)
        if self.frames <= 2 * self.max_font_size:
            self.font_size -= 0.5 * dt
        self.frames -= dt

    # helper method for ship Destroyed
    def use_a_life(self):
        Score.delLife()
        while self.frames > 0:
            dt = fpsClock.tick(FPS) * REFERENCE_FPS / 1000.0
            if self.is_max():
                freezeScreen = self.screen.subsurface(
                    pygame.Rect(0, 0, WIDTH, HEIGHT)
                ).copy()
                self.shipDestroyed(freezeScreen, dt)
            else:
                self.shipDestroyed(freezeScreen, dt)
            pygame.display.update()

    def title_banner(self, dt=1):
        font_diff = max(1, int(self.max_font_size - self.font_size + 1))
        textBlit(
            self.screen,
            "ASTEROIDS",
            "Arial",
            font_diff,
            BLUE,
            "center",
            winWidth / 2,
            winHeight / 8
            + font_diff
            * winHeight
            / (15 * self.max_font_size),
        )
        if self.frames > 2 * self.max_font_size:
            self.font_size -= 0.5 * dt
        self.frames -= dt

    def start_up(self):
        global Space
        infoSurf = self.info_blit(False, True)
        starting_up = True
        run = True
        ship = Ship(self.screen)
        bullets = pygame.sprite.Group()  # set up some groups
        rocks = pygame.sprite.Group()
        burst = Burst()

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
                        print("ship is at", ship.p.x, ship.p.y)
                elif event.type == KEYUP:
                    if event.key == K_1:
                        Space.set_space(1, 1)
                        del ship
                        ship = Ship(self.screen)
                    elif event.key == K_2:
                        Space.set_space(1, -1)
                        infoSurf = self.info_blit(False, True)
                        del ship
                        ship = Ship(self.screen)
                    elif event.key == K_3:
                        Space.set_space(-1, 1)
                        infoSurf = self.info_blit(False, True)
                        del ship
                        ship = Ship(self.screen)
                    elif event.key == K_4:
                        Space.set_space(-1, -1)
                        infoSurf = self.info_blit(False, True)
                        del ship
                        ship = Ship(self.screen)
                    elif event.key == K_c:
                        run = False
                    elif event.key == K_SPACE:
                        Bullet(self.screen, ship, bullets)
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

            self.screen.fill(WHITE)

            if starting_up and self.frames > 0:
                self.title_banner(dt)
            else:
                starting_up = False

            if not starting_up:
                self.screen.blit(infoSurf, (0, 0, winWidth, winHeight))

            ship.update(thrust, left, right, dt)

            if burst.update(dt):
                Bullet(self.screen, ship, bullets)

            bullets.update(bullets, rocks, dt)

            pygame.display.update()

        self.reset()
        bullets.empty()
        rocks.empty()
        del ship

    def info_blit(self, pause=False, screen_shot=False):
        if screen_shot:
            self.screen.fill(WHITE)
        textBlit(
            self.screen,
            "Controls",
            "Arial",
            80,
            BLUE,
            "center",
            winWidth / 2,
            4 * winHeight / 24,
        )
        textBlit(
            self.screen,
            "Use <left> and <right> to burn the",
            "Arial",
            40,
            BLUE,
            "center",
            winWidth / 2,
            6.5 * winHeight / 24,
        )
        textBlit(
            self.screen,
            "right or left engine,",
            "Arial",
            40,
            BLUE,
            "center",
            winWidth / 2,
            8 * winHeight / 24,
        )
        textBlit(
            self.screen,
            "<up> to burn both engines",
            "Arial",
            40,
            BLUE,
            "center",
            winWidth / 2,
            9.5 * winHeight / 24,
        )
        textBlit(
            self.screen,
            "<space> (or <b>urst) to shoot",
            "Arial",
            40,
            BLUE,
            "center",
            winWidth / 2,
            14 * winHeight / 24,
        )
        if pause:
            textBlit(
                self.screen,
                "<p>ause",
                "Arial",
                40,
                BLUE,
                "center",
                winWidth / 2,
                16 * winHeight / 24,
            )
            textBlit(
                self.screen,
                "<space> to continue",
                "Arial",
                40,
                BLUE,
                "center",
                winWidth / 2,
                17.5 * winHeight / 24,
            )
            textBlit(
                self.screen,
                "<q>uit game",
                "Arial",
                40,
                BLUE,
                "center",
                winWidth / 2,
                19 * winHeight / 24,
            )
            textBlit(
                self.screen,
                "hit <space> or <q>",
                "Arial",
                40,
                BLUE,
                "center",
                winWidth / 2,
                21 * winHeight / 24,
            )
            textBlit(
                self.screen,
                "PAUSED",
                "Arial",
                60,
                BLUE,
                "center",
                winWidth / 2,
                12 * winHeight / 24,
            )
        else:
            textBlit(
                self.screen,
                "<p>ause to show CONTROLS",
                "Arial",
                40,
                BLUE,
                "center",
                winWidth / 2,
                15.5 * winHeight / 24,
            )
            textBlit(
                self.screen,
                "Choose a space: <1> torus (default)",
                "Arial",
                40,
                RED,
                "center",
                winWidth / 2,
                17 * winHeight / 24,
            )
            textBlit(
                self.screen,
                "<2> or <3> Klein bottle, <4> projective plane",
                "Arial",
                40,
                RED,
                "center",
                winWidth / 2,
                18.5 * winHeight / 24,
            )
            if Space.t_x == 1 and Space.t_y == 1:
                textBlit(
                    self.screen,
                    "space is the torus",
                    "Arial",
                    40,
                    BLUE,
                    "center",
                    winWidth / 2,
                    20 * winHeight / 24,
                )
            if Space.t_x == 1 and Space.t_y == -1:
                textBlit(
                    self.screen,
                    "Klein bottle, twist connecting left and right",
                    "Arial",
                    40,
                    BLUE,
                    "center",
                    winWidth / 2,
                    20 * winHeight / 24,
                )
            if Space.t_x == -1 and Space.t_y == 1:
                textBlit(
                    self.screen,
                    "Klein bottle, twist connecting top and bottom",
                    "Arial",
                    40,
                    BLUE,
                    "center",
                    winWidth / 2,
                    20 * winHeight / 24,
                )
            if Space.t_x == -1 and Space.t_y == -1:
                textBlit(
                    self.screen,
                    "space is the projective plane",
                    "Arial",
                    40,
                    BLUE,
                    "center",
                    winWidth / 2,
                    20 * winHeight / 24,
                )
            textBlit(
                self.screen,
                "fly around and shoot, then hit <c>ontinue",
                "Arial",
                40,
                RED,
                "center",
                winWidth / 2,
                21.5 * winHeight / 24,
            )
        if screen_shot:
            return self.screen.subsurface(pygame.Rect(0, 0, WIDTH, HEIGHT)).copy()

    # this freezes the game and screen
    def info(self, pause):
        self.info_blit(pause)
        paused = True

        while paused:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYUP and event.key == K_q):
                    pygame.quit()
                    sys.exit()
                elif event.type == KEYUP and event.key == K_SPACE:
                    return False

            pygame.display.update()
            fpsClock.tick(FPS)

    def reset(self):
        self.frames = self.frames_max
        self.font_size = self.max_font_size
        self._life_given = False

    def is_max(self):
        return self.frames == self.frames_max

    def lose(self):

        textBlit(
            self.screen,
            "No lives left!",
            "Arial",
            80,
            BLUE,
            "center",
            winWidth / 2,
            winHeight / 3,
        )
        textBlit(
            self.screen,
            "<r>estart  or  <q>uit",
            "Arial",
            50,
            BLUE,
            "center",
            winWidth / 2,
            2 * winHeight / 3,
        )
        textBlit(
            self.screen,
            "Final score: " + str(Score.get()),
            "Arial",
            80,
            RED,
            "center",
            winWidth / 2,
            winHeight / 2,
        )

        paused = True

        while paused:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYUP and event.key == K_q):
                    pygame.quit()
                    sys.exit()
                elif event.type == KEYUP and event.key == K_r:
                    paused = False
                    break

            pygame.display.update()
            fpsClock.tick(FPS)


class Burst:
    def __init__(self, num=5, delay=20, shoot=False):
        self.num = num
        self.delay = delay  # reference frames between shots
        self.shoot = shoot
        self.shots_left = num
        self.timer = self.delay  # fires immediately on first update

    def update(self, dt):
        """Returns True when a bullet should be fired."""
        if not self.shoot:
            return False
        if self.shots_left <= 0:
            self.reset()
            return False
        self.timer += dt
        if self.timer >= self.delay:
            self.timer -= self.delay
            self.shots_left -= 1
            return True
        return False

    def reset(self):
        self.shots_left = self.num
        self.timer = self.delay  # fires immediately on next burst
        self.shoot = False


def main():

    global fpsClock, Score

    pygame.init()
    fpsClock = pygame.time.Clock()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Asteroids")
    pygame.mouse.set_visible(0)

    fader = Fader(screen)

    fader.start_up()  # show a startup animation and CONTROLS

    ship = Ship(screen)  # create a Ship

    # set up some groups
    bullets = pygame.sprite.Group()
    rocks = pygame.sprite.Group()

    burst = Burst()  # rapid fire

    starting_up = True
    pause = False

    while len(rocks) < NUM_ROCKS:  # add some rocks
        BigRock(screen, rocks)

    num_rocks = NUM_ROCKS

    while True:
        dt = fpsClock.tick(FPS) * REFERENCE_FPS / 1000.0
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                if event.key == K_b:
                    burst.shoot = True
                if event.key == K_SPACE:
                    Bullet(screen, ship, bullets)
            elif event.type == KEYUP:
                if event.key == K_q:
                    pygame.quit()
                    sys.exit()
                if event.key == K_p:
                    pause = True
        thrust = left = right = False
        keys = pygame.key.get_pressed()
        if keys[K_UP]:
            thrust = True
        if keys[K_LEFT]:
            left = True
        if keys[K_RIGHT]:
            right = True

        screen.fill(WHITE)

        ship.update(thrust, left, right, dt)

        if burst.update(dt):
            Bullet(screen, ship, bullets)

        if bullets:
            bullets.update(bullets, rocks, dt)
        rocks.update(dt)

        if pygame.sprite.spritecollideany(
            ship, rocks, pygame.sprite.collide_circle_ratio(0.7)
        ):
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

        Score.draw(screen, rocks)

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


main()
