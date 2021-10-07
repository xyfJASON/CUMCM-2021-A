import numpy as np
from scipy.optimize import minimize

from constant import const
from readdata import get_dis_pairs, get_layers


def solve_with_minimize(mu: np.ndarray,
                        base_pos: np.ndarray,
                        v: np.ndarray,
                        movable_ids: np.ndarray):
    base_d = get_dis_pairs(base_pos)[:, 2]
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

    # def fun_with_constr(lamb):
    #     loss = np.sum((expand(lamb) - mu[movable_ids]) ** 2)
    #     real_pos = base_pos.copy()
    #     real_pos[movable_ids] += expand(lamb).reshape(-1, 1) * v[movable_ids]
    #     real_d = get_dis_pairs(real_pos)[:, 2]
    #     loss += ((np.fabs(real_d - base_d) / base_d - 0.0007) ** 2).sum()
    #     return loss

    def fun_without_constr(lamb):
        assert lamb.shape == (17, )
        loss = np.sum((expand(lamb) - mu[movable_ids]) ** 2)  # L2
        # tmpjac = 2 * (expand(lamb) - mu[movable_ids])
        # jac, p = [], 0
        # for length in lengths:
        #     jac.append(tmpjac[p:p+length].sum())
        # jac = np.array(jac)
        # assert jac.shape == (17, )
        return loss

    def constr(lamb):
        assert lamb.shape == (17, )
        real_pos = base_pos.copy()
        real_pos[movable_ids] += expand(lamb).reshape(-1, 1) * v[movable_ids]
        real_d = get_dis_pairs(real_pos)[:, 2]
        assert real_d.shape == (const.n_edges, )
        return real_d - 0.9993 * base_d, 1.0007 * base_d - real_d

    # bounds = [(-0.6, 0.6)] * 17
    constraints = [
        dict(type='ineq', fun=lambda x: constr(x)[0]),
        dict(type='ineq', fun=lambda x: constr(x)[1]),
        dict(type='ineq', fun=lambda x: 0.6 - x),
        dict(type='ineq', fun=lambda x: x + 0.6)
    ]

    # noinspection PyTypeChecker
    res = minimize(fun=fun_without_constr,
                   # jac=True,
                   x0=np.zeros(17),
                   method='COBYLA',
                   # bounds=bounds,
                   constraints=constraints,
                   tol=1e-6,
                   options=dict(
                       disp=True,
                       # maxfun=1500000,
                   ),
                   )
    assert res.success
    return res['x'], res['fun']


def solve_with_minimize_all(mu: np.ndarray,
                            base_pos: np.ndarray,
                            v: np.ndarray,
                            movable_ids: np.ndarray):
    import warnings
    warnings.warn('`minimize_all` will stuck on this problem!')
    base_d = get_dis_pairs(base_pos)[:, 2]

    def fun_without_constr(lamb):
        assert lamb.shape == (len(movable_ids), )
        loss = np.sum((lamb - mu[movable_ids]) ** 2)  # L2
        jac = 2 * (lamb - mu[movable_ids])
        return loss, jac

    def constr(lamb):
        assert lamb.shape == (len(movable_ids), )
        real_pos = base_pos.copy()
        real_pos[movable_ids] += lamb.reshape(-1, 1) * v[movable_ids]
        real_d = get_dis_pairs(real_pos)[:, 2]
        assert real_d.shape == (const.n_edges, )
        return real_d - 0.9993 * base_d, 1.0007 * base_d - real_d

    bounds = [(-0.6, 0.6)] * len(movable_ids)
    constraints = [
        dict(type='ineq', fun=lambda x: constr(x)[0]),
        dict(type='ineq', fun=lambda x: constr(x)[1]),
        # dict(type='ineq', fun=lambda x: 0.6 - x),
        # dict(type='ineq', fun=lambda x: x + 0.6)
    ]

    # noinspection PyTypeChecker
    res = minimize(fun=fun_without_constr,
                   jac=True,
                   x0=np.zeros(len(movable_ids)),
                   method='SLSQP',
                   bounds=bounds,
                   constraints=constraints,
                   tol=1e-6,
                   options=dict(
                       disp=True,
                       # maxfun=1500000,
                   ),
                   )
    assert res.success
    return res['x'], res['fun']
