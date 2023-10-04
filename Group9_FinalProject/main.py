from GridWorld import GridWorld
from Agent import Agent
import matplotlib.pyplot as plt
import keyboard 



# possible actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

def main():
    grid_env = GridWorld(Agent(), 10, 10)
    grid_env.visualize_grid()

    while True:  # making a loop
        try:  # used try so that if user pressed other than the given key error will not be shown
            if keyboard.is_pressed('q'):  # if key 'q' is pressed 
                print('quit')
                break  # finishing the loop
            
            if keyboard.is_pressed('i'):
                grid_env.step(UP)
                grid_env.visualize_grid()

            if keyboard.is_pressed('l'):
                grid_env.step(RIGHT)
                grid_env.visualize_grid()
            
            if keyboard.is_pressed('k'):
                grid_env.step(DOWN)
                grid_env.visualize_grid()

            if keyboard.is_pressed('j'):
                grid_env.step(LEFT)
                grid_env.visualize_grid()
        except:
            break  # if user pressed a key other than the given key the loop will break

if __name__ == "__main__":
    main()