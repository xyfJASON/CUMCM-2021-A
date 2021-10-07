import numpy as np


def rotate_matrix(alpha, beta):
    sa = np.sin(alpha)
    ca = np.cos(alpha)
    sb = np.sin(beta)
    cb = np.cos(beta)
    M = np.array([
        [sa, -ca, 0],
        [ca*sb, sa*sb, -cb],
        [ca*cb, sa*cb, sb]
    ]).T
    M = np.linalg.inv(M)
    return M


class const:
    R = 300.4  # 球面半径
    D = 500  # 球面口径
    n_points = 2226
    n_edges = 6525
    n_faces = 4300
    F = 0.466 * R  # 两球面半径差
    r_receiver = 0.5  # 馈源舱接收半径
    d_parab = 300  # 抛物面口径
    rangep = (2*(R-F-301), 2*(R-F-299.8))  # 参数 p 理论范围（实际取 [-285, -275]）
    alpha = 36.795 / 180 * np.pi
    beta = 78.169 / 180 * np.pi

    M = rotate_matrix(alpha, beta)  # 坐标变换矩阵
