import numpy as np
import random
from tqdm import tqdm

from constant import const


def calc_base_rate():
    cnt = 0
    tot = 1000000
    for _ in tqdm(range(tot)):
        rho = random.random() * 150
        alpha = random.random() * 2 * np.pi
        x = rho * np.cos(alpha)
        y = rho * np.sin(alpha)
        z = -np.sqrt(const.R ** 2 - x ** 2 - y ** 2)
        mat = np.array([[0, -z, y],
                        [-y, x, 0],
                        [x, y, z]])
        a, b, c = (np.linalg.inv(mat) @ np.array([-y, 0, z]).reshape(-1, 1)).flatten()
        X = x + a / c * (const.F - const.R - z)
        Y = y + b / c * (const.F - const.R - z)
        if X**2 + Y**2 <= 0.5**2:
            cnt += 1
    print('Base rate is: %.4f%%' % (cnt / tot * 100))


if __name__ == '__main__':
    calc_base_rate()
