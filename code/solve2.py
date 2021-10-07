import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from readdata import get_final_data, get_ideal_pos, get_dis_pairs, get_layers
from constant import const
from solve_with_qp import solve_with_qp
from solve_with_minimize import solve_with_minimize, solve_with_minimize_all
from visualize import visualize_result


def check_constr(lamb, base_pos, v, movable_ids):
    base_d = get_dis_pairs(base_pos)[:, 2]
    assert base_d.shape == (const.n_edges, )
    real_pos = base_pos.copy()
    real_pos[movable_ids] += lamb.reshape(-1, 1) * v[movable_ids]
    real_d = get_dis_pairs(real_pos)[:, 2]
    assert real_d.shape == (const.n_edges, )

    bad_mask = (np.fabs(real_d - base_d) / base_d) > 0.0007
    print('There are %d bad edges.' % bad_mask.sum())


def solve_for_one_p(p: np.ndarray,
                    method: str,
                    t_number: int):
    assert method in ['minimize', 'qp', 'minimize_all']
    base_pos, v = get_final_data(t_number)
    movable_mask = np.sum(base_pos[:, :2] ** 2, axis=1) <= (150 ** 2)
    movable_ids = np.arange(const.n_points)[movable_mask]
    mu, _ = get_ideal_pos(p[0], base_pos, v)
    if method == 'minimize':
        lamb, loss = solve_with_minimize(mu, base_pos, v, movable_ids)
    elif method == 'minimize_all':
        lamb, loss = solve_with_minimize_all(mu, base_pos, v, movable_ids)
    else:
        lamb, loss = solve_with_qp(mu, base_pos, v, movable_ids)
    return loss, lamb


def solve(method: str,
          t_number: int,
          test_p: float):
    # if test_p != 0, test with it; or run with minimize
    assert method in ['minimize', 'qp', 'minimize_all']
    if t_number == 2:
        assert method == 'qp'

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

        def expand(lamb) -> np.ndarray:
            assert lamb.shape == (17, )
            exlamb = np.concatenate([np.tile(lamb[i], r) for i, r in enumerate([len(l) for l in layers])])
            assert exlamb.shape == (len(movable_ids), )
            return exlamb

    if test_p == 0:
        def fun(x):
            return solve_for_one_p(x, method, t_number)[0]
        res = minimize(fun=fun,
                       x0=np.array([-280.5]),
                       options=dict(
                           disp=True,
                       ))
        print(res)
        exit()
    # T1(cvxopt): -280.76746957
    # T2(cvxopt): -280.75523667
    # T1(minimize): -280.78486814

    best_p = test_p
    loss, best_lamb = solve_for_one_p(np.array([best_p]), method, t_number)
    if t_number == 1 and method == 'minimize':
        # noinspection PyUnboundLocalVariable
        best_lamb = expand(best_lamb)
    print('loss is', loss)
    np.save('best_loss.npy', loss)
    np.save('best_p.npy', best_p)
    np.save('best_lamb.npy', best_lamb)
    real_pos = base_pos.copy()
    real_pos[movable_ids] += best_lamb.reshape(-1, 1) * v[movable_ids]
    np.save('best_pos.npy', real_pos)

    check_constr(best_lamb, base_pos, v, movable_ids)
    best_mu, ideal_pos = get_ideal_pos(best_p, base_pos, v)
    assert (np.fabs(best_lamb - best_mu[movable_ids]) <= 0.6).all()
    visualize_result(best_lamb, base_pos, v, movable_ids, show=True, save=False, rotate=(t_number == 2))


if __name__ == '__main__':
    solve(method='qp',
          t_number=2,
          test_p=-280.75523667)
