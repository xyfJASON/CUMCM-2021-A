import numpy as np
from cvxopt.solvers import qp
from cvxopt import matrix

from readdata import get_dis_pairs


def solve_with_qp(mu: np.ndarray,
                  base_pos: np.ndarray,
                  movable_ids: np.ndarray,
                  clip: float):
    assert clip <= 0.6
    m = len(movable_ids)
    func = {idx: i for i, idx in enumerate(movable_ids)}
    P = np.eye(m) * 2
    q = -2 * mu[movable_ids]
    G, h = [], []
    adj_pairs = get_dis_pairs(base_pos)
    for adj in adj_pairs:
        if adj[0] not in func and adj[1] not in func:
            continue

        tmp = np.zeros(m)
        if adj[0] in func:
            tmp[func[adj[0]]] = 1
        if adj[1] in func:
            tmp[func[adj[1]]] = -1
        G.append(tmp); h.append(np.sqrt(0.0014) * adj[2])

        tmp = np.zeros(m)
        if adj[1] in func:
            tmp[func[adj[1]]] = 1
        if adj[0] in func:
            tmp[func[adj[0]]] = -1
        G.append(tmp); h.append(np.sqrt(0.0014) * adj[2])

    G = np.concatenate((np.eye(m), np.eye(m) * -1, G), axis=0)
    h = np.concatenate((np.ones(2*m) * clip, h), axis=0)

    res = qp(matrix(P), matrix(q), matrix(G), matrix(h), initvals=np.zeros(m))
    lamb = np.array(res['x']).flatten()
    return lamb, np.sum((lamb - mu[movable_ids]) ** 2)
