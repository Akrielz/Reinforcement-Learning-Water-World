from constants import *

class Molecule:

    def __init__(self, x, y, status, radius=RADIUS, speed=SPEED):
        self.x = x
        self.y = y
        self.r = radius
        self.alpha_dir = np.random.random_sample()*MAX_DEGREE
        self.status = status
        self.speed = speed

    def update(self, is_player = False):
        nx = self.x + self.speed * np.math.cos(self.alpha_dir)
        ny = self.y + self.speed * np.math.sin(self.alpha_dir)

        needs_x_update = False
        if ny - self.r < 0 or ny + self.r >= GAME_HEIGHT:
            if is_player == False:
                self.alpha_dir = 2*PI - self.alpha_dir

        if nx - self.r < 0 or nx + self.r >= GAME_WIDTH:
            if is_player == False:
                self.alpha_dir = 3*PI - self.alpha_dir

        self.x += self.speed * np.math.cos(self.alpha_dir)
        self.y += self.speed * np.math.sin(self.alpha_dir)

        if is_player:
            self.x = (self.x + GAME_WIDTH) % GAME_WIDTH
        if is_player:
            self.y = (self.y + GAME_HEIGHT) % GAME_HEIGHT

    def distance_to(self, other):
        return ((self.x-other.x)**2 + (self.y-other.y)**2)**(0.5)

    def is_coliding(self, other):
        return self.distance_to(other) < self.r + other.r


class Sensor(Molecule):
    def __init__(self, x, y, status, radius=RADIUS, speed=SPEED):
        super().__init__(x, y, status, radius, speed)
        self.detected = NOTHING_DETECTED