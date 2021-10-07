import pandas as pd
import numpy as np

from constant import const


def get_ori_data(i: int):
    return pd.read_csv('./data/%d.csv' % i, encoding='GBK')


def get_map_data(i: int):
    return np.load('./data/map%d.npy' % i)


def get_adj_pairs():
    return np.load('./data/adj_pairs.npy')


def get_movable_ids(t_number: int):
    base_pos, v = get_final_data(t_number)
    movable_mask = np.sum(base_pos[:, :2] ** 2, axis=1) <= (150 ** 2)
    movable_ids = np.arange(const.n_points)[movable_mask]
    return movable_ids


def get_layers():
    mapping = {d[0]: i for i, d in enumerate(get_ori_data(1).to_numpy())}
    data = get_ori_data(1).to_numpy()
    layers = []
    last_char = 'E'
    for i, d in enumerate(data):
        if last_char == 'E' and d[0][0] == 'A':
            layers.append([])
        layers[-1].append(mapping[d[0]])
        last_char = d[0][0]
    return layers


def get_final_data(t: int):
    if t == 1:
        p = np.load('./data/final_p.npy')
        v = np.load('./data/final_v.npy')
    else:
        p = np.load('./data/final_p2.npy')
        v = np.load('./data/final_v2.npy')
    assert p.shape == v.shape == (const.n_points, 3)
    return p, v


def get_dis_pairs(pos: np.ndarray):
    """
    :param pos: (n, 3), 主索节点移动后的坐标
    :return: (k, 3), 相邻节点之间的距离, [i, j, dis_ij]
    """
    adj_pairs = get_adj_pairs()
    assert adj_pairs.shape == (const.n_edges, 2)
    dis = pos[adj_pairs[:, 0]] - pos[adj_pairs[:, 1]]
    dis = np.linalg.norm(dis, axis=1)
    res = np.concatenate((adj_pairs, dis.reshape(-1, 1)), axis=1)
    assert res.shape == (const.n_edges, 3)
    return res


def get_ideal_pos(p: float, base_pos: np.ndarray, v: np.ndarray):
    """
    :param p: 抛物面参数
    :param base_pos: 基准面坐标
    :param v: 主索节点方向向量
    :return: 主索节点按移动方向到抛物面上的移动距离，及移动后的坐标
    """
    c = - p * p + 2 * (const.R - const.F) * p
    x0, y0, z0 = base_pos[1:, 0], base_pos[1:, 1], base_pos[1:, 2]
    i, j, k = v[1:, 0], v[1:, 1], v[1:, 2]
    Delta = (
            (p**2)*(k**2) + 2*i*j*x0*y0 + 2*j*k*p*y0 + 2*i*k*p*x0
            - (j**2)*(x0**2) - (i**2)*(y0**2)
            - 2*p*z0*(i**2) - 2*p*z0*(j**2)
            - c*(i**2) - c*(j**2)
    )
    mu1 = (-(i*x0+j*y0+p*k) + np.sqrt(Delta)) / (i**2 + j**2)
    mu2 = (-(i*x0+j*y0+p*k) - np.sqrt(Delta)) / (i**2 + j**2)
    x0, y0, z0 = base_pos[0]
    i, j, k = v[0]
    mu1 = np.concatenate(([-(x0**2 + y0**2 + 2*p*z0 + c) / (2*p*k)], mu1))
    mu2 = np.concatenate(([-(x0**2 + y0**2 + 2*p*z0 + c) / (2*p*k)], mu2))
    mu1[np.fabs(mu1) > np.fabs(mu2)] = mu2[np.fabs(mu1) > np.fabs(mu2)]
    return mu1, base_pos + mu1.reshape(-1, 1) * v


# if __name__ == '__main__':
#     base_pos, v = get_final_data(1)
#     movable_mask = np.sum(base_pos[:, :2] ** 2, axis=1) <= (150 ** 2)
#     movable_ids = np.arange(const.n_points)[movable_mask]
#     mu, _ = get_ideal_pos(-280.76746957, base_pos, v)
#     print(mu[movable_ids])
#     pos = base_pos.copy()
#     import matplotlib.pyplot as plt
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     ax.scatter3D(pos[movable_ids][:, 0], pos[movable_ids][:, 1], mu[movable_ids], s=4)
#     plt.show()
