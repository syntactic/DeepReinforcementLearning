import pygame

class GameWindow():
    def __init__(self, width:int=500, height:int=500):
        self.width = width
        self.height = height

        pygame.init()
        pygame.font.init() # allow font for text writing
        self.screen = pygame.display.set_mode([self.width, self.height])
    
    def draw(self, im):
        # convert the image to a surface to draw on the screen
        im = pygame.surfarray.make_surface(im)
        # draw the surface representing the board on the screen
        self.screen.blit(im, (0,0))

    def check_quit(self):
        playing = True
        for event in pygame.event.get():
            # quit if the window is closed
            if event.type == pygame.QUIT:
                playing = False
        return not playing
    
    def flip(self):
        # flip the display
        pygame.display.flip()
