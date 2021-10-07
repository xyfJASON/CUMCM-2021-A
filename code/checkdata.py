import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from readdata import get_final_data, get_map_data, get_dis_pairs, get_adj_pairs, get_ori_data, get_movable_ids
from constant import const


def check_for_on_sphere():
    suo = get_map_data(1)
    abserr = np.fabs((300.4 ** 2) - (suo ** 2).sum(axis=1))
    print(abserr)


def check_for_alignment():
    mapping = {d[0]: i for i, d in enumerate(get_ori_data(1).to_numpy())}
    inv_mapping = {v: k for k, v in mapping.items()}

    suo = get_map_data(1)  # 主索节点坐标
    xia = get_map_data(2)[:, :3]  # 下端点坐标
    shang = get_map_data(2)[:, 3:]  # 上端点坐标
    v1 = shang - xia
    v2 = suo - shang
    v3 = suo - xia
    vo = -suo
    cos_sim1o = np.diagonal(cosine_similarity(v1, vo))
    cos_sim2o = np.diagonal(cosine_similarity(v2, vo))
    cos_sim3o = np.diagonal(cosine_similarity(v3, vo))
    print(np.min(cos_sim1o))
    print(np.min(cos_sim2o))
    print(np.min(cos_sim3o))
    print(np.argsort(cos_sim1o)[:10])
    print(np.argsort(cos_sim2o)[:10])
    print(np.argsort(cos_sim3o)[:10])

    abnorm_id = np.argsort(cos_sim2o)[:10]
    cos_sim = np.sort(cos_sim2o)
    print(np.arccos(cos_sim[:10]) * 180 / np.pi)
    print([inv_mapping[idx] for idx in abnorm_id])

    plt.scatter(suo[:, 0], suo[:, 1], s=1, label='normal')
    plt.scatter(suo[abnorm_id, 0], suo[abnorm_id, 1], color='r', s=4, label='abnormal')
    plt.axis('equal')
    plt.xlabel('$x(m)$')
    plt.ylabel('$y(m)$')
    plt.title('主索节点 $x-y$ 投影图')
    plt.legend()
    plt.show()


def plot_data_on_xy(data, ax, title='', label=''):
    ax.scatter(data[:, 0], data[:, 1], s=1, label=label)
    ax.axis('equal')
    ax.set_xlabel('$x(m)$')
    ax.set_ylabel('$y(m)$')
    ax.set_title(title)
    ax.legend()


def plot_xy():
    suo = get_map_data(1)  # 主索节点坐标
    xia = get_map_data(2)[:, :3]  # 下端点坐标
    shang = get_map_data(2)[:, 3:]  # 上端点坐标
    fig, ax = plt.subplots(1, 3)
    plot_data_on_xy(suo, ax[0], '主索节点 $x-y$ 投影图')
    plot_data_on_xy(xia, ax[1], '促动器下端点 $x-y$ 投影图')
    plot_data_on_xy(shang, ax[2], '基准态促动器上端点 $x-y$ 投影图')
    plt.show()


def check_final_data():
    p, v = get_final_data(1)
    assert p.shape == v.shape == (const.n_points, 3)
    adj_pairs = get_adj_pairs()
    minsim = 1
    for i in range(len(p)):
        x, y = np.nonzero(adj_pairs == i)
        y = 1 - y
        minsim = min(minsim, (np.min(cosine_similarity(v[i:i+1], v[adj_pairs[x, y]]))))
    print(minsim)


def check_distance_to_center():
    p, v = get_final_data(1)
    assert p.shape == v.shape == (const.n_points, 3)
    dists = np.sum(p[:, :2] ** 2, axis=1)
    # dists = p[:, 2]
    fig, ax = plt.subplots(1, 1)
    ax.scatter(np.arange(const.n_points), dists, s=4)
    plt.show()


def calc_adj_theta():
    adj_pairs = get_adj_pairs()
    p, v = get_final_data(1)
    hudu, du = [], []
    dis = []
    for adj in adj_pairs:
        cossim = cosine_similarity(v[adj[0]].reshape(1, -1),
                                   v[adj[1]].reshape(1, -1))[0, 0]
        hudu.append(np.arccos(cossim))
        du.append(np.arccos(cossim) * 180 / np.pi)
        dis.append(np.linalg.norm(p[adj[0]] - p[adj[1]]))
    hudu, du = np.array(hudu), np.array(du)
    dis = np.array(dis)
    print(hudu.mean(), hudu.std(), np.max(hudu), np.min(hudu))
    print(du.mean(), du.std(), np.max(du), np.min(du))
    print(dis.mean(), dis.std(), np.max(dis), np.min(dis))
    # 结论：
    # 相邻节点伸缩方向 0.03 +- 0.0025 弧度，最大 0.0557 弧度
    # 相邻节点伸缩方向 2.17 +- 0.1439 度，最大 3.19 度
    # 相邻节点距离 11.391 +- 0.728 m


def check_pairs():
    p, v = get_final_data(1)
    adj_pairs = get_adj_pairs()
    dis_pairs = get_dis_pairs(p)
    print(np.all(adj_pairs == dis_pairs[:, :2]))
    p, v = get_final_data(2)
    dis_pairs = get_dis_pairs(p)
    print(np.all(adj_pairs == dis_pairs[:, :2]))


def check_coor_rotate():
    orip, oriv = get_final_data(1)
    rotp, rotv = get_final_data(2)
    print('rotate max error:', np.max(np.fabs((const.M @ orip.T).T - rotp)))
    print('rotate max error:', np.max(np.fabs((const.M @ oriv.T).T - rotv)))
    print('inverse rotate max error:', np.max(np.fabs((np.linalg.inv(const.M) @ rotp.T).T - orip)))
    print('inverse rotate max error:', np.max(np.fabs((np.linalg.inv(const.M) @ rotv.T).T - oriv)))


def plot_curve():
    fig, ax = plt.subplots(1, 3)
    xrange = np.linspace(-100, 100, 100)
    circ = -np.sqrt((const.R ** 2 - xrange ** 2))
    p = np.array([-283, -280, -277])
    c = 2 * (const.R - const.F) * p - p * p
    for i in range(3):
        ax[i].plot(xrange, circ, c='grey', label='球面截面')
        ax[i].plot(xrange, (-c[i]-xrange**2)/p[i]/2, label='理想抛物面截面')
        ax[i].set_title('p=%f' % p[i])
        ax[i].set_xlabel('$x(m)$')
        ax[i].set_ylabel('$z(m)$')
        ax[i].axis('equal')
        ax[i].legend()
    plt.show()


def check_result():
    import pandas as pd
    res1 = pd.read_csv('./result/result1.csv', header=None)
    res2 = pd.read_csv('./result/result2.csv', header=None)
    res3 = pd.read_csv('./result/result3.csv', header=None)

    dingdian = res1.to_numpy()
    print((const.M @ dingdian.T).T)

    jiedian = res2.to_numpy()
    mapping = {d[0]: i for i, d in enumerate(get_ori_data(1).to_numpy())}
    print((get_movable_ids(2) == np.array([mapping[i[0]] for i in jiedian])).all())
    std = np.load('./result/quadratic/T2/best_pos.npy')
    std = (np.linalg.inv(const.M) @ std.T).T
    print((std[get_movable_ids(2)]-jiedian[:, 1:].astype(float)).min())

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(std[:, 0], std[:, 1], std[:, 2], c='grey', alpha=0.1, s=2)
    ax.scatter(jiedian[:, 1].astype(float), jiedian[:, 2].astype(float), jiedian[:, 3].astype(float), s=2)
    ax.set_xlim(-250, 250)
    ax.set_ylim(-250, 250)
    ax.set_zlim(-500, 0)
    plt.show()

    shensuo = res3.to_numpy()
    lamb = np.load('./result/quadratic/T2/best_lamb.npy')
    assert len(shensuo) == len(lamb) == 692
    print((shensuo[:, 1].astype(float) - lamb).min())


if __name__ == '__main__':
    check_result()
