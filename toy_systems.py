import matplotlib.pyplot as plt
import matplotlib.animation as ani
from systems import ToySystem

from settings import system_str, fps, interval, frames, save_gif

if __name__ == "__main__":
    toy = ToySystem(system_str)

    if save_gif:
        repeat = False
    else:
        frames = None
        repeat = None
    anim = ani.FuncAnimation(toy.fig, toy.update, interval=interval, blit=True, frames=frames, repeat=repeat)
    plt.show()

    if save_gif:
        filename = system_str+'.gif'
        writergif = ani.PillowWriter(fps=fps)
        anim.save(filename, writer=writergif)
