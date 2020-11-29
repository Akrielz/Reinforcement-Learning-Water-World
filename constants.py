import numpy as np

# Screen Constants
GAME_WIDTH = 1366
GAME_HEIGHT = 768

OFFSET = 100

# Frame Constants
NR_MAX_FRAMES = 100

# Color Constants
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Molecule Constants
PI = np.math.pi
RADIUS = 30
R = RADIUS
MAX_DEGREE = 2*PI
SPEED = 30

# Molecule States 
PLAYER = 0
ALLY = 1
ENEMY = -1

# Molecule Numbers
NR_ENEMIES_INIT = 0
NR_ALLIES_INIT = 10

# Spawn Chance Enemy
ENEMY_CHANCE = 0.5