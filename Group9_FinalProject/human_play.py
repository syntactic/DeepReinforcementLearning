from GridWorld import GridWorld
from Agent import Agent
import pygame

# possible actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

def main():

    # initialize the grid environment
    grid_env = GridWorld(Agent(), 10, 10)
    win_state = grid_env.win_state

    ### set up display
    pygame.init()
    pygame.font.init() # allow font for text writing
    screen = pygame.display.set_mode([500, 500])

    playing = True
    game_won = False

    # game loop     
    while playing:    
           
        ### handle game events
        for event in pygame.event.get():      # pygame.event.get()
            # quit if the window is closed
            if event.type == pygame.QUIT:
                playing = False
            
            # if the event is a keypress
            if event.type == pygame.KEYDOWN and not game_won:
                if event.key == pygame.K_UP:
                    grid_env.step(UP)
                if event.key == pygame.K_RIGHT:
                    grid_env.step(RIGHT)
                if event.key == pygame.K_DOWN:
                    grid_env.step(DOWN)
                if event.key == pygame.K_LEFT:
                    grid_env.step(LEFT)
        ###
    
        ### Draw the grid state to the game screen

        # get an image of the current state of the grid (and upsample 50x to fill game space)
        im = grid_env.grid_image().repeat(50, axis=0).repeat(50, axis=1)

        # convert the image to a surface to draw on the screen
        im = pygame.surfarray.make_surface(im)

        # draw the surface representing the board on the screen
        screen.blit(im, (0,0))
        ###

        ### check if the player has completed the game
        curr_agent_pos = grid_env.agent.pos; 
        if (curr_agent_pos.x == win_state.x) and \
            (curr_agent_pos.y == win_state.y):
            game_won = True
            my_font = pygame.font.SysFont('Comic Sans MS', 30)
            text_surface = my_font.render('Game won! Press "s" to save', False, (0, 00, 0))
            screen.blit(text_surface, (20,10))
        ###


        # flip the display
        pygame.display.flip()

if __name__ == "__main__":
    main()