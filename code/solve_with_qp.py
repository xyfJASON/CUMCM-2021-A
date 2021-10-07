import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from cvxopt.solvers import qp
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

from readdata import get_dis_pairs, get_adj_pairs
from constant import const


def construct_adj_mat(base_pos: np.ndarray,
                      movable_ids: np.ndarray):
    func = {idx: i for i, idx in enumerate(movable_ids)}
    G, h = [], []
    adj_pairs = get_dis_pairs(base_pos)
    for adj in adj_pairs:
        if adj[0] not in func and adj[1] not in func:
            continue

        tmp = np.zeros(len(movable_ids))
        if adj[0] in func:
            tmp[func[adj[0]]] = 1
        if adj[1] in func:
            tmp[func[adj[1]]] = -1
        G.append(tmp); h.append(np.sqrt(0.0014) * adj[2])

        tmp = np.zeros(len(movable_ids))
        if adj[1] in func:
            tmp[func[adj[1]]] = 1
        if adj[0] in func:
            tmp[func[adj[0]]] = -1
        G.append(tmp); h.append(np.sqrt(0.0014) * adj[2])
    return G, h


def check(lamb: np.ndarray,
          base_pos: np.ndarray,
          v: np.ndarray,
          movable_ids: np.ndarray,
          mam: dict) -> np.ndarray:
    """
    :return: ids (in movables_ids) not satisfy constraint
    """
    real_pos = base_pos.copy()
    real_pos[movable_ids] += lamb.reshape(-1, 1) * v[movable_ids]
    real_dis_pairs = get_dis_pairs(real_pos)[:, 2]
    base_dis_pairs = get_dis_pairs(base_pos)[:, 2]
    mask = np.fabs(real_dis_pairs - base_dis_pairs) / base_dis_pairs > 0.0007
    adj_pairs = get_adj_pairs()
    assert mask.shape[0] == adj_pairs.shape[0] == const.n_edges
    bad_ids = list(set(adj_pairs[mask].flatten()))
    bad_ids = np.array([mam[idx] for idx in bad_ids if idx in movable_ids])
    return bad_ids


def solve_with_qp(mu: np.ndarray,
                  base_pos: np.ndarray,
                  v: np.ndarray,
                  movable_ids: np.ndarray):
    m = len(movable_ids)
    mam = {idx: i for i, idx in enumerate(movable_ids)}

    P = np.eye(m) * 2
    q = -2 * mu[movable_ids]
    G_adj, h_adj = construct_adj_mat(base_pos, movable_ids)
    G = np.concatenate((np.eye(m), np.eye(m) * -1, G_adj), axis=0)
    # G = np.concatenate((np.eye(m), np.eye(m) * -1), axis=0)
    clip = np.ones(m) * 0.6 * (0.96 ** 25)

    while True:
        h = np.concatenate((np.ones(m) * clip, np.ones(m) * clip, h_adj), axis=0)
        # h = np.concatenate((np.ones(m) * clip, np.ones(m) * clip), axis=0)
        res = qp(matrix(P), matrix(q), matrix(G), matrix(h), initvals=np.zeros(m))
        lamb = np.array(res['x']).flatten()
        bad_ids = check(lamb, base_pos, v, movable_ids, mam)
        if bad_ids.size > 0:
            print('\t %d bad points' % bad_ids.size)
            clip[bad_ids] *= 0.98  # TODO
            continue
        else:
            break

    return lamb, np.sum((lamb - mu[movable_ids]) ** 2)
