from __future__ import absolute_import

import numpy as np
from scipy.optimize import OptimizeResult

import ipopt
from .jacobian import FunctionWithApproxJacobian


class IpoptProblemWrapper(object):
    def __init__(self, fun, args=(), jac=None, hess=None, hessp=None,
                 constraints=(), eps=1e-8):
        if hess is not None or hessp is not None:
            raise NotImplementedError('Using hessian matrixes is not yet implemented!')
        if jac is None:
            fun = FunctionWithApproxJacobian(fun, epsilon=eps, verbose=False)
            jac = fun.jac
        elif not callable(jac):
            raise NotImplementedError('For now, the jacobian has to be provided!')
        self.fun = fun
        self.jac = jac
        self.args = args
        self._constraint_funs = []
        self._constraint_jacs = []
        self._constraint_args = []
        if isinstance(constraints, dict):
            constraints = (constraints, )
        for con in constraints:
            con_fun = con['fun']
            con_jac = con.get('jac', None)
            if con_jac is None:
                con_fun = FunctionWithApproxJacobian(con_fun, epsilon=eps, verbose=False)
                con_jac = con_fun.jac
            con_args = con.get('args', [])
            self._constraint_funs.append(con_fun)
            self._constraint_jacs.append(con_jac)
            self._constraint_args.append(con_args)

    def objective(self, x):
        return self.fun(x, *self.args)

    def gradient(self, x):
        return self.jac(x, *self.args)  # .T

    def constraints(self, x):
        con_values = []
        for fun, arg in zip(self._constraint_funs, self._constraint_args):
            con_values.append(fun(x, *self.args))
        return np.hstack(con_values)

    def jacobian(self, x):
        con_values = []
        for fun, arg in zip(self._constraint_jacs, self._constraint_args):
            con_values.append(fun(x, *self.args))
        return np.vstack(con_values)


def get_bounds(bounds):
    if bounds is None:
        return None, None
    else:
        lb = [b[0] for b in bounds]
        ub = [b[1] for b in bounds]
        return lb, ub


def get_constraint_bounds(constraints, x0):
    if isinstance(constraints, dict):
        constraints = (constraints, )
    cl = []
    cu = []
    if isinstance(constraints, dict):
        constraints = (constraints, )
    for con in constraints:
        m = len(np.atleast_1d(con['fun'](x0, *con.get('args', []))))
        cl.extend(np.zeros(m))
        if con['type'] == 'eq':
            cu.extend(np.zeros(m))
        elif con['type'] == 'ineq':
            cu.extend(1e19*np.ones(m))
        else:
            raise ValueError(con['type'])
    cl = np.array(cl)
    cu = np.array(cu)

    return cl, cu


def minimize_ipopt(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None,
                   bounds=None, constraints=(), tol=None, callback=None, options=None):

    _x0 = np.atleast_1d(x0)
    problem = IpoptProblemWrapper(fun, args=args, jac=jac, hess=hess,
                                  hessp=hessp, constraints=constraints)
    lb, ub = get_bounds(bounds)

    cl, cu = get_constraint_bounds(constraints, x0)

    if options is None:
        options = {}

    #ipopt.setLoggingLevel(options.get('disp', logging.FATAL))

    nlp = ipopt.problem(n = len(_x0),
                        m = len(cl),
                        problem_obj=problem,
                        lb=lb,
                        ub=ub,
                        cl=cl,
                        cu=cu)
    if 'disp' in options:
        options['print_level'] = options.pop('disp')
    elif 'print_level' not in options:
        options['print_level'] = 0
    if not 'tol' in options:
        options['tol'] = tol or 1e-8
    if not 'mu_strategy' in options:
        options['mu_strategy'] = 'adaptive'
    if not 'hessian_approximation' in options:
        if hess is None and hessp is None:
            options['hessian_approximation'] = 'limited-memory'
    for option, value in options.items():
        nlp.addOption(option, value)

    x, info = nlp.solve(_x0)

    if np.asarray(x0).shape == ():
        x = x[0]

    # TODO: How to get the iteration count etc?
    return OptimizeResult(x=x, success=info['status'] == 0, status=info['status'],
                          message=info['status_msg'],
                          fun=info['obj_val'],
                          info=info)
