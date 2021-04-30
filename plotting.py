import matplotlib.pyplot as plt


def setup_plot(title_str):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axis('square')
    ax.autoscale(False)
    ax.axis([-1.4, 1.4, -1.4, 1.4])
    ax.axis('off')
    ax.set_title(title_str)
    return fig, ax


def make_style_kwargs(p1, p2, lw, style='base'):
    style_kwargs1 = dict(p1=p1, p2=p2, lw=lw)
    if style == 'base':
        style_kwargs2 = dict(color='darkgray', alpha=1.0, zorder=10)
    elif style == 'accent1':
        style_kwargs2 = dict(color='tab:blue', alpha=1.0, zorder=20)
    elif style == 'accent2':
        style_kwargs2 = dict(color='tab:red', alpha=1.0, zorder=20)
    else:
        raise ValueError
    return {**style_kwargs1, **style_kwargs2}


def make_line(line=None, ax=None, p1=None, p2=None, **style_kwargs):
    if line is None:
        line, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **style_kwargs)
    else:
        line.set_xdata([p1[0], p2[0]])
        line.set_ydata([p1[1], p2[1]])
    return line


def mod_artist_props(d_list, size_scale=1.0, alpha_scale=1.0, zorder_offset=0):
    d_list_new = []
    for d in d_list:
        d_new = {key: value for key, value in d.items()}
        d_new['lw'] *= size_scale
        d_new['alpha'] *= alpha_scale
        d_new['zorder'] += zorder_offset
        d_list_new.append(d_new)
    return d_list_new


def draw_system_parametric(x, u, x_eq, u_eq, ax, artists, make_artist_props):
    d_list = make_artist_props(x, u)
    d_list = mod_artist_props(d_list, zorder_offset=100)
    d_list_eq = make_artist_props(x_eq, u_eq)
    d_list_eq = mod_artist_props(d_list_eq, size_scale=1.0, alpha_scale=0.3)
    if artists is None:
        artists = [None for d in d_list]
    return [make_line(line, ax, **d) for line, d in zip(artists+artists, d_list+d_list_eq)]

