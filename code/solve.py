import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from readdata import get_final_data, get_ideal_pos, get_dis_pairs, get_layers
from constant import const
from visualize import visualize_result, visualize_loss_p
from solve_with_qp import solve_with_qp
from solve_with_minimize import solve_with_minimize


def check_constr(lamb, base_pos, v, movable_ids):
    base_d = get_dis_pairs(base_pos)[:, 2]
    assert base_d.shape == (const.n_edges, )
    real_pos = base_pos.copy()
    real_pos[movable_ids] += lamb.reshape(-1, 1) * v[movable_ids]
    real_d = get_dis_pairs(real_pos)[:, 2]
    assert real_d.shape == (const.n_edges, )

    bad_mask = (np.fabs(real_d - base_d) / base_d) > 0.0007
    print('There are %d bad edges.' % bad_mask.sum())


def solve_for_one_p(p: float,
                    method: str,
                    t_number: int):
    assert method in ['minimize', 'qp']
    base_pos, v = get_final_data(t_number)
    movable_mask = np.sum(base_pos[:, :2] ** 2, axis=1) <= (150 ** 2)
    movable_ids = np.arange(const.n_points)[movable_mask]
    mu, _ = get_ideal_pos(p, base_pos, v)
    if method == 'minimize':
        lamb, loss = solve_with_minimize(mu, base_pos, v, movable_ids)
    else:
        lamb, loss = solve_with_qp(mu, base_pos, v, movable_ids)
    return loss, lamb


def solve(method: str,
          rangep: np.ndarray,
          t_number: int):
    assert method in ['minimize', 'qp']

    base_pos, v = get_final_data(t_number)
    movable_mask = np.sum(base_pos[:, :2] ** 2, axis=1) <= (150 ** 2)
    movable_ids = np.arange(const.n_points)[movable_mask]

    if t_number == 1 and method == 'minimize':
        layers = get_layers()
        tmp = []
        for layer in layers:
            t = [l for l in layer if l in movable_ids]
            if len(t) != 0:
                tmp.append(t)
        layers = tmp
        lengths = [len(l) for l in layers]
        assert len(layers) == len(lengths) == 17

        def expand(_lamb) -> np.ndarray:
            assert _lamb.shape == (17, )
            exlamb = np.concatenate([np.tile(_lamb[i], r) for i, r in enumerate([len(l) for l in layers])])
            assert exlamb.shape == (len(movable_ids), )
            return exlamb

    history = []
    for p in rangep:
        print('==> solve for p=%f' % p)
        loss, lamb = solve_for_one_p(p, method, t_number)
        history.append((loss, lamb, p))

    minid = np.argmin([h[0] for h in history])
    best_lamb = history[minid][1]
    best_p = history[minid][2]

    if t_number == 1 and method == 'minimize':
        # noinspection PyUnboundLocalVariable
        best_lamb = expand(best_lamb)

    print('best p is', best_p)
    best_coo = base_pos.copy()
    best_coo[movable_ids] += best_lamb.reshape(-1, 1) * v[movable_ids]
    np.save('result.npy', np.array((best_coo, best_lamb, best_p, history), dtype=object))

    check_constr(best_lamb, base_pos, v, movable_ids)
    best_mu, ideal_pos = get_ideal_pos(best_p, base_pos, v)
    assert (np.fabs(best_lamb - best_mu[movable_ids]) <= 0.6).all()
    visualize_result(best_lamb, base_pos, v, movable_ids, show=True, save=False, rotate=(t_number == 2))
    visualize_loss_p([h[2] for h in history], [h[0] for h in history], show=True, save=False)


if __name__ == '__main__':
    solve(method='qp',
          rangep=np.linspace(-280.7, -275, 1),
          t_number=1)
