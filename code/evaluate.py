import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from constant import const
from readdata import get_map_data, get_final_data, get_ideal_pos, get_movable_ids
from visualize import visualize_result


def calc_area(tri: np.ndarray) -> float:
    assert tri.shape == (3, 2)
    v1 = tri[0] - tri[1]
    v2 = tri[0] - tri[2]
    return np.fabs(v1[0] * v2[1] - v1[1] * v2[0]) / 2.


def reflect(tri: np.ndarray) -> tuple[float, np.ndarray]:
    v1 = tri[0] - tri[1]
    v2 = tri[0] - tri[2]
    n = np.cross(v1, v2)
    n = -n if n[2] < 0 else n
    n /= np.linalg.norm(n)
    cos2theta = 2 * (n[2] ** 2) - 1
    N = np.array([[0, n[2], -n[1]],
                  [-n[2], 0, n[0]],
                  [-n[0], -n[1], -n[2]]])
    invN = np.linalg.inv(N)
    ftri = np.empty((3, 2))
    for i in range(3):
        u = np.array([0, 0, -1])
        v = invN @ np.concatenate((np.cross(u, n)[:2], [np.dot(u, n)]))
        ftri[i, 0] = tri[i, 0] + (const.F - const.R - tri[i, 2]) / v[2] * v[0]
        ftri[i, 1] = tri[i, 1] + (const.F - const.R - tri[i, 2]) / v[2] * v[1]
    return cos2theta, ftri


def evaluate(cur_pos: np.ndarray, counted_id: np.ndarray, r: float = 0.5) -> float:
    assert cur_pos.shape == (const.n_points, 3)
    tri = get_map_data(3)
    assert tri.shape == (const.n_faces, 3)
    assert cur_pos[tri].shape == (const.n_faces, 3, 3)

    ftris, cos2thetas = [], []
    for t in tri:
        # t point ID of a triangle (3, )
        # cur_pos[t]: (3, 3), each row is a (x, y, z) coord of a triangle
        if (t[0] not in counted_id) or (t[1] not in counted_id) or (t[2] not in counted_id):
            continue
        cos2theta, ftri = reflect(cur_pos[t])  # ftri (3, 2) 是馈源舱平面光线坐标
        ftris.append(ftri.flatten())
        cos2thetas.append(cos2theta)
    ftris = np.array(ftris)

    with open('evaluate_in.txt', 'w') as f:
        f.write('%.6f\n' % r)
        for ftri in ftris:
            f.write('%.6f %.6f %.6f %.6f %.6f %.6f\n' % tuple(ftri.tolist()))

    with os.popen('./calc_intersection') as f:
        inter = f.readlines()[0]
    inter_area = list(map(float, inter.split()))

    area_inter, area_tri = 0., 0.
    for i, (ftri, cos2theta) in enumerate(zip(ftris, cos2thetas)):
        area_inter += inter_area[i] * cos2theta
        area_tri += calc_area(ftri.reshape(3, 2)) * cos2theta
    return area_inter / area_tri


def evaluate_base(t_number: int, r: float = 0.5):
    base_pos, v = get_final_data(t_number)
    movable_ids = get_movable_ids(t_number)
    rate = evaluate(base_pos, movable_ids, r)
    print('Receive rate is %.6f%%' % (rate * 100))
    return rate


def evaluate_ideal(p: float, t_number: int, r: float = 0.5):
    base_pos, v = get_final_data(t_number)
    movable_ids = get_movable_ids(t_number)
    real_pos = get_ideal_pos(p, base_pos, v)[1]
    rate = evaluate(real_pos, movable_ids, r)
    print('Receive rate is %.6f%%' % (rate * 100))
    return rate


def evaluate_real(real_pos, t_number: int, r: float = 0.5):
    base_pos, _ = get_final_data(t_number)
    movable_ids = get_movable_ids(t_number)
    rate = evaluate(real_pos, movable_ids, r)
    print('Receive rate is %.6f%%' % (rate * 100))
    return rate


def T1(method: str):
    best_pos = np.load('./result/' + method + '/T1/best_pos.npy')
    best_lamb = np.load('./result/' + method + '/T1/best_lamb.npy')
    best_p = np.load('./result/' + method + '/T1/best_p.npy')
    best_loss = np.load('./result/' + method + '/T1/best_loss.npy')
    print('p=', best_p)
    print('c=', 2*best_p*(const.R-const.F)-best_p*best_p)
    print('loss=', best_loss)
    evaluate_real(real_pos=best_pos, t_number=1)
    evaluate_ideal(p=best_p, t_number=1)
    base_pos, v = get_final_data(1)
    movable_ids = get_movable_ids(1)

    visualize_result(best_lamb, base_pos, v, movable_ids, show=True, save=False)
    pos = base_pos.copy()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(pos[movable_ids][:, 0], pos[movable_ids][:, 1], best_lamb, s=4)
    ax.set_title('形变图')
    plt.show()


def T2(r: float = 0.5):
    best_pos = np.load('./result/quadratic/T2/best_pos.npy')
    best_lamb = np.load('./result/quadratic/T2/best_lamb.npy')
    best_p = np.load('./result/quadratic/T2/best_p.npy')
    best_loss = np.load('./result/quadratic/T2/best_loss.npy')
    print('p=', best_p)
    print('c=', 2*best_p*(const.R-const.F)-best_p*best_p)
    print('loss=', best_loss)
    evaluate_real(real_pos=best_pos, t_number=2, r=r)
    evaluate_ideal(p=best_p, t_number=2, r=r)
    base_pos, v = get_final_data(2)
    movable_ids = get_movable_ids(2)

    visualize_result(best_lamb, base_pos, v, movable_ids, show=True, save=False, rotate=True)
    pos = base_pos.copy()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(pos[movable_ids][:, 0], pos[movable_ids][:, 1], best_lamb, s=4)
    ax.set_title('形变图')
    plt.show()


def plot_rate_r():
    best_pos = np.load('./result/quadratic/T2/best_pos.npy')
    best_p = np.load('./result/quadratic/T2/best_p.npy')
    best_loss = np.load('./result/quadratic/T2/best_loss.npy')
    print('p=', best_p)
    print('c=', 2*best_p*(const.R-const.F)-best_p*best_p)
    print('loss=', best_loss)

    real, ideal, base = [], [], []
    ranger = np.linspace(0.5, 10, 100)
    for r in ranger:
        real.append(evaluate_real(real_pos=best_pos, t_number=2, r=r))
        ideal.append(evaluate_ideal(p=best_p, t_number=2, r=r))
        base.append(evaluate_base(t_number=2, r=r))
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(ranger, real, label='real', c='b')
    ax[0].plot(ranger, ideal, label='ideal', c='r')
    ax[0].plot(ranger, base, label='base', c='g')
    ax[0].set_xlabel('馈源舱半径(m)')
    ax[0].set_ylabel('接收比')
    ax[0].legend()
    ax[1].plot(ranger, np.array(real) / np.array(base), label='real/base', c='b')
    ax[1].plot(ranger, np.array(ideal) / np.array(base), label='ideal/base', c='r')
    ax[1].set_xlabel('馈源舱半径(m)')
    ax[1].set_ylabel('相对接收比')
    ax[1].legend()
    plt.show()


if __name__ == '__main__':
    T1(method='quadratic')
    # T1(method='minimize')
    # T2(r=0.5)
    # evaluate_base(t_number=2)

    # plot_rate_r()
