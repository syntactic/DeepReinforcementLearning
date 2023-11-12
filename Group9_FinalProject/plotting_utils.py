import numpy as np
import matplotlib.pyplot as plt

import matplotlib
import matplotlib as mpl
from matplotlib import colors
import matplotlib.animation as animation
plt.rcParams['svg.fonttype'] = 'none'

from utils import Buffer, Position, index_of_value_in_2d_array
from constants import *
from GridWorld import GridWorld

# discrete color map to interpret the grid: floor, wall, agent, win
cmap = colors.ListedColormap(['#CACACA', '#625151', '#E53A3A', '#FFE400'])

def heatmap(data, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    modified from matplotlib.org source
    source: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Remove x and y ticks/labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.axis('off')

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    modified from matplotlib.org source
    source: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), fontsize = 14, **kw)
            texts.append(text)

    return texts

def save_fig(fig, path):
    plt.tight_layout()
    fig.savefig(path + '.svg', format='svg', dpi=1200, bbox_inches='tight')

def plot_heatmap(data, path="", valfmt="{x:.2f}", save=False):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    im, cbar = heatmap(data=data, ax=ax)
    texts = annotate_heatmap(im, valfmt=valfmt)

    if save:
        save_fig(fig, path)
    plt.show()
    plt.clf()
    plt.close()


def plot_values_over_index(data, path = "", filename="graph", xlabel='x', ylabel='y', figsize=(8, 4), save=False):
    # set up the figure and axes
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    ax.plot(range(len(data)), data)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if save:
        save_fig(fig, path)

    plt.show()
    plt.clf()
    plt.close()

def plot_state_visitation_map(state_list, dones = None, name="", save=False, save_path=""):
    """ plots a heatmap of visitation freq by state for a state_list taken 
        from a player's trajectories """
    assert len(state_list) > 0

    # get shape of state from first state
    s = state_list[0]

    # initialize visitaion map
    visitation_map = np.zeros(s.shape)
    
    # loop through each state to update histogram of visits
    i = 0
    for s in state_list:
        visitation_map[s==PLAYER] += 1
        if dones:
            if dones[i]:
                visitation_map[s==WIN] += 1
        i+=1
    plot_heatmap(data=visitation_map, valfmt="{x:.0f}",
                 path = save_path + name+ "_visitation_map", save=save)
    
def plot_trajectories(state_list, dones, name="", save=False, save_path=""):
    """ plots line plots of trajectories over the grid """
    assert len(state_list) > 0

    fig, ax = plt.subplots(figsize = (7, 7))

    # plot the grid as an image
    im_state = np.copy(state_list[0])
    im_state[im_state==PLAYER] = FLOOR
    #ax.imshow(im_state, cmap=cmap)
    ax.imshow(np.zeros((10,10)))
    plt.axis('off')

    # split states by games
    win_x, win_y = index_of_value_in_2d_array(im_state, WIN)
    win_pos = Position(win_x, win_y)

    # separate the trajectories into lists
    i = 0
    trajs = [[]]
    for j in range(len(state_list)):
        x, y = index_of_value_in_2d_array(state_list[j], PLAYER)
        trajs[i].append((x,y))
        if dones[j]:
            trajs[i].append((win_x, win_y))
            trajs.append([])
            i+=1
    
    for i in range(len(trajs)):
        x = [trajs[i][j][0] for j in range(len(trajs[i]))]
        y = [trajs[i][j][1] for j in range(len(trajs[i]))]
        ax.plot(y, x)

    
def gen_play_video(state_list, name="example", save_path=""):
    fig = plt.figure()

    ims = []
    for i in range(len(state_list)):
        im = plt.imshow(state_list[i], cmap=cmap, animated=True)
        ims.append([im])
        plt.axis('off')
        plt.tight_layout()

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)
    
    writergif = animation.PillowWriter(fps=12)

    ani.save(save_path+name+'.gif', writergif)
            


    
    
    
    

    