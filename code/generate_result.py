import numpy as np
import pandas as pd

from readdata import get_ori_data, get_final_data
from constant import const

base_pos, v = get_final_data(2)
movable_mask = np.sum(base_pos[:, :2] ** 2, axis=1) <= (150 ** 2)
movable_ids = np.arange(const.n_points)[movable_mask]


def restore(pos: np.ndarray) -> np.ndarray:
    assert pos.shape[1] == 3
    return (np.linalg.inv(const.M) @ pos.T).T


def gen1():
    p = np.load('./result/quadratic/T2/best_p.npy')
    c = 2 * (const.R - const.F) * p - p * p
    vertex = np.array([[0, 0, -c / (2 * p)]])
    vertex = restore(vertex)
    df = pd.DataFrame(vertex)
    # noinspection PyTypeChecker
    df.to_csv('result/result1.csv', header=False, index=False, float_format='%.10f')


def gen2():
    pos = np.load('./result/quadratic/T2/best_pos.npy')
    ids = np.array([d[0] for d in get_ori_data(1).to_numpy()])[:, np.newaxis]
    points = restore(pos)
    df = pd.DataFrame(np.concatenate((ids[movable_ids], points[movable_ids]), axis=1))
    # noinspection PyTypeChecker
    df.to_csv('result/result2.csv', header=False, index=False, float_format='%.10f')


def gen3():
    lamb = np.load('./result/quadratic/T2/best_lamb.npy')
    ids = np.array([d[0] for d in get_ori_data(1).to_numpy()])[:, np.newaxis]
    df = pd.DataFrame(np.concatenate((ids[movable_ids], lamb[:, np.newaxis]), axis=1))
    # noinspection PyTypeChecker
    df.to_csv('result/result3.csv', header=False, index=False, float_format='%.10f')


if __name__ == '__main__':
    gen1()
    gen2()
    gen3()
