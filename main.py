import pygame
from nn_model import *
import copy

molecules = None 
score = None
nr_allies_left = None

neural_network = Neural_Network()

def render_molecule(screen, molecules):
    for m in molecules:
        color = None
        if m.status == ENEMY:
            color = RED
        elif m.status == ALLY:
            color = GREEN
        elif m.status == PLAYER:
            color = BLUE

        pygame.draw.circle(screen, color, (m.x, m.y), m.r)


def eval_position(action, molecules_updated, in_game_score):
    # molecules copied
    sum = 0

    molecules_updated[-1].alpha_dir = action/360*2*PI
    molecules_updated[-1].update()

    for i, m in enumerate(molecules[:-1]):
        if molecules[-1].is_coliding(m):
            if m.status == ENEMY:
                sum -= 1
            else:
                sum += 1

    sum += in_game_score
    for i, m in enumerate(molecules_updated[:-1]):
        sum += 5*m.r * m.status / molecules_updated[-1].distance_to(m)

    print(sum)
    return sum


def init_new_game():
    global molecules, score, nr_allies_left, neural_network
    molecules = []
    for _ in range(NR_ENEMIES_INIT + NR_ALLIES_INIT + 1):
        x = np.random.random_sample() * (GAME_WIDTH - OFFSET) + OFFSET / 2
        y = np.random.random_sample() * (GAME_HEIGHT - OFFSET) + OFFSET / 2

        status = None
        if _ < NR_ENEMIES_INIT:
            status = ENEMY
        elif _ < NR_ALLIES_INIT + NR_ENEMIES_INIT:
            status = ALLY
        else:
            status = PLAYER
        molecule = Molecule(x=x, y=y, status=status)
        molecules.append(molecule)

    score = 0
    nr_allies_left = NR_ALLIES_INIT

pygame.init()
font = pygame.font.Font('freesansbold.ttf', 32)
screen = pygame.display.set_mode([GAME_WIDTH, GAME_HEIGHT])
init_new_game()

running = True
frame = 0
while running:

    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the background with white - TO CLEAR THE BACKGROUND
    screen.fill(WHITE)

    # Draw
    text = font.render("Score: " + str(score), True, BLACK, WHITE)
    textRect = text.get_rect() 
    textRect.center = (100, 50)
    screen.blit(text, textRect)

    text = font.render("Frame: " + str(frame), True, BLACK, WHITE)
    textRect = text.get_rect() 
    textRect.center = (GAME_WIDTH-100, 50)
    screen.blit(text, textRect)

    render_molecule(screen, molecules)

    input_values = [molecules[-1].x, molecules[-1].y, molecules[-1].speed]
    for i, m in enumerate(molecules[:-1]):
        input_values.append(m.x)
        input_values.append(m.y)
        input_values.append(m.r)
        input_values.append(m.status)
        input_values.append(m.speed)
        input_values.append(m.alpha_dir)
        input_values.append(molecules[-1].distance_to(m))
    
    for i, m in enumerate(molecules[:-1]):
        # m.update()
        pass
    
    action = neural_network.feed(input_values, eval_func=eval_position, molecules=copy.deepcopy(molecules), in_game_score = score)
    molecules[-1].alpha_dir = action/360*2*PI
    molecules[-1].update()

    for i, m in enumerate(molecules[:-1]):
        if molecules[-1].is_coliding(m):
            if m.status == ENEMY:
                score -= 1
            else:
                score += 1
                nr_allies_left -= 1

            x = np.random.random_sample() * (GAME_WIDTH - OFFSET) + OFFSET / 2
            y = np.random.random_sample() * (GAME_HEIGHT - OFFSET) + OFFSET / 2
            status = ENEMY if np.random.random_sample() < ENEMY_CHANCE or m.status == ENEMY else ALLY
            
            if status == ALLY:
                nr_allies_left += 1

            molecules[i] = Molecule(x=x, y=y, status=status)

    # Flip the display
    pygame.display.update()

    if nr_allies_left == 0 or frame >= NR_MAX_FRAMES:
        print(score)
        neural_network.update()
        init_new_game()
        frame = 0

    frame += 1

pygame.quit()