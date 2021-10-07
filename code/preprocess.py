import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from readdata import get_ori_data, get_map_data, get_final_data
from constant import const

mapping = {d[0]: i for i, d in enumerate(get_ori_data(1).to_numpy())}
inv_mapping = {v: k for k, v in mapping.items()}

from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
sorted_names = [name for hsv, name in by_hsv]


def show_layers():
    data = get_ori_data(1).to_numpy()
    layers = []
    last_char = 'E'
    for d in data:
        if last_char == 'E' and d[0][0] == 'A':
            layers.append([])
        layers[-1].append(d[1:].astype(float))
        last_char = d[0][0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, layer in enumerate(layers):
        pos = np.array(layer)
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                   s=4, c=colors[sorted_names[i*5]], label='layer %d' % i)
    ax.set_xlim(-250, 250)
    ax.set_ylim(-250, 250)
    ax.set_zlim(-400, 0)
    ax.set_xlabel('$x(m)$')
    ax.set_ylabel('$y(m)$')
    ax.set_zlabel('$z(m)$')
    ax.set_title('数据的分层特征')
    plt.show()


def show_groups():
    data = get_ori_data(1).to_numpy()
    p, v = get_final_data(1)
    groupA = np.array([mapping[data[i, 0]] for i in range(const.n_points) if data[i, 0][0] == 'A'])
    groupB = np.array([mapping[data[i, 0]] for i in range(const.n_points) if data[i, 0][0] == 'B'])
    groupC = np.array([mapping[data[i, 0]] for i in range(const.n_points) if data[i, 0][0] == 'C'])
    groupD = np.array([mapping[data[i, 0]] for i in range(const.n_points) if data[i, 0][0] == 'D'])
    groupE = np.array([mapping[data[i, 0]] for i in range(const.n_points) if data[i, 0][0] == 'E'])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(p[groupA, 0], p[groupA, 1], p[groupA, 2], s=4, c='r', label='A')
    ax.scatter(p[groupB, 0], p[groupB, 1], p[groupB, 2], s=4, c='g', label='B')
    ax.scatter(p[groupC, 0], p[groupC, 1], p[groupC, 2], s=4, c='b', label='C')
    ax.scatter(p[groupD, 0], p[groupD, 1], p[groupD, 2], s=4, c='orange', label='D')
    ax.scatter(p[groupE, 0], p[groupE, 1], p[groupE, 2], s=4, c='purple', label='E')
    ax.set_xlim(-250, 250)
    ax.set_ylim(-250, 250)
    ax.set_zlim(-400, 0)
    ax.set_xlabel('$x(m)$')
    ax.set_ylabel('$y(m)$')
    ax.set_zlabel('$z(m)$')
    ax.set_title('数据的分区特征')
    ax.legend()
    plt.show()


def gen_mapping_data(save=False):
    data1 = get_ori_data(1).to_numpy()
    data2 = get_ori_data(2).to_numpy()
    data3 = get_ori_data(3).to_numpy()
    data1 = data1[:, 1:]
    data2 = data2[:, 1:]
    data3 = np.array([
        [mapping[data3[i, 0]],
         mapping[data3[i, 1]],
         mapping[data3[i, 2]]]
        for i in range(len(data3))])
    if save:
        np.save('map1.npy', data1.astype(float))
        np.save('map2.npy', data2.astype(float))
        np.save('map3.npy', data3.astype(int))
    print(data1)
    print(data2)
    print(data3)


def gen_adj_pairs(save=False):
    """
    :return: (n, 2), each row contain a pair of adjacent points
    """
    data_tri = get_map_data(3)
    n, m = data_tri.shape
    s = set()
    for i in range(n):
        for j in range(m):
            u = data_tri[i, j]
            v = data_tri[i, (j+1) % m]
            s.add((min(u, v), max(u, v)))
    adj_pairs = np.array([list(e) for e in s])
    if save:
        np.save('adj_pairs.npy', adj_pairs)
    print(adj_pairs)


def gen_final_data(save=False):
    coo = get_map_data(1)
    xia = get_map_data(2)[:, :3]
    shang = get_map_data(2)[:, 3:]
    v = shang - xia
    v = v / np.linalg.norm(v, axis=1).reshape(-1, 1)
    if save:
        np.save('final_p.npy', coo)
        np.save('final_v.npy', v)
    print(coo)
    print(v)


if __name__ == '__main__':
    show_groups()
