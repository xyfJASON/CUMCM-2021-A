import numpy as np
import matplotlib.pyplot as plt

from constant import const


def visualize_result(lamb, base_pos, v, movable_ids, show=True, save=True, rotate=False):
    if rotate:
        base_pos = (np.linalg.inv(const.M) @ base_pos.T).T
        v = (np.linalg.inv(const.M) @ v.T).T
    show_pos = base_pos.copy()
    show_amplify = 100
    show_pos[movable_ids] += lamb.reshape(-1, 1) * show_amplify * v[movable_ids]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(base_pos[:, 0], base_pos[:, 1], base_pos[:, 2],
                 s=2, color='grey', label='base', alpha=0.3)
    ax.scatter3D(show_pos[movable_ids][:, 0],
                 show_pos[movable_ids][:, 1],
                 show_pos[movable_ids][:, 2],
                 s=2, color='r', label='real')

    # show_pos = base_pos + mu.reshape(-1, 1) * show_amplify * v
    # ax.scatter3D(show_pos[:, 0],
    #              show_pos[:, 1],
    #              show_pos[:, 2],
    #              s=2, color='b', label='ideal', alpha=0.5)

    ax.set_xlim(-250, 250)
    ax.set_ylim(-250, 250)
    ax.set_zlim(-400, 0)
    ax.set_title('变形效果放大 %d 倍图像' % show_amplify)
    ax.set_xlabel('$x(m)$')
    ax.set_ylabel('$y(m)$')
    ax.set_zlabel('$z(m)$')
    ax.legend()
    if save:
        plt.savefig('result.png', dpi=300)
    if show:
        plt.show()


def visualize_loss_p(p, loss, show=True, save=True):
    fig, ax = plt.subplots(1, 1)
    ax.plot(p, loss)
    ax.set_title('loss~p')
    ax.set_xlabel('p')
    ax.set_ylabel('loss')
    if save:
        plt.savefig('loss-p.png', dpi=300)
    if show:
        plt.show()
