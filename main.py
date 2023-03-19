import matplotlib.pyplot as plt
import matplotlib.animation as ani

from systems import ToySystem
from settings import system_str, fps, interval, frames, repeat, save_gif

if __name__ == "__main__":
    toy = ToySystem(system_str)

    anim = ani.FuncAnimation(toy.fig, toy.update, interval=interval, blit=True, frames=frames, repeat=repeat)
    plt.show()

    if save_gif:
        filename = system_str+'.gif'
        writer = ani.PillowWriter(fps=fps)
        anim.save(filename, writer=writer)
