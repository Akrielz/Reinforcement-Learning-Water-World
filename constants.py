import numpy as np

# Screen Constants
GAME_WIDTH = 1366
GAME_HEIGHT = 768
GAME_MAX_DISTANCE = (GAME_HEIGHT**2 + GAME_WIDTH**2)**(1/2)

OFFSET = 100

# Visual Constants
VISUAL_ACTIVATED = True
SLEEP_DELAY = 0 #1/120

# Frame Constants
NR_MAX_FRAMES = 1000

# Color Constants
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Molecule Constants
PI = np.math.pi
RADIUS = 30
MAXIMUM_RADIUS = 50
R = RADIUS
MAX_DEGREE = 2*PI
SPEED = 10

# Molecule States 
PLAYER = 0
ALLY = 1
ENEMY = -1

# Molecule Numbers
NR_ENEMIES_INIT = 7
NR_ALLIES_INIT = 7

# Spawn Chance Enemy
ENEMY_CHANCE = 0.5

# Neural Network Strats
NR_INPUTS = 4+9*(NR_ENEMIES_INIT+NR_ALLIES_INIT)
NR_ACTIONS = 16

# Neural Network State
TEST_MODE = False
TEST_MODEL_ID = "153_last"
CONTINUE_MODEL = False

# Checking Rule
NR_GEN_CHECK = 50
