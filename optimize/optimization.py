"""
Author: Matthias Kuemmerer, 2014

Some wrappers around scipy.optimize.minimize to make optimization
of functions with multiple parameters easier
"""


import sys
import numpy as np
import scipy.optimize


class FunctionWithApproxJacobian(object):
    def __init__(self, func, epsilon, verbose=True):
        self._func = func
        self.epsilon = epsilon
        self.value_cache = {}
        self.verbose = verbose

    def __call__(self, x, *args, **kwargs):
        key = tuple(x)
        if not key in self.value_cache:
            self.log('.')
            value = self._func(x, *args, **kwargs)
            if np.any(np.isnan(value)):
                print "Warning! nan function value encountered at {0}".format(x)
            self.value_cache[key] = value
        return self.value_cache[key]

    def func(self, x, *args, **kwargs):
        print x
        return self(x, *args, **kwargs)

    def log(self, msg):
        if self.verbose:
            sys.stdout.write(msg)
            sys.stdout.flush()

    def jac(self, x, *args, **kwargs):
        self.log('G[')
        x0 = np.asfarray(x)
        #print x0
        dxs = np.zeros((len(x0), len(x0) + 1))
        for i in range(len(x0)):
            dxs[i, i + 1] = self.epsilon
        results = [self(*(x0 + dxs[:, i], ) + args, **kwargs) for i in range(len(x0) + 1)]
        jac = np.zeros([len(x0), len(np.atleast_1d(results[0]))])
        for i in range(len(x0)):
            jac[i] = (results[i + 1] - results[0]) / self.epsilon
        self.log(']')
        return jac.transpose()


class FunctionWithApproxJacobianCentral(FunctionWithApproxJacobian):
    def jac(self, x, *args, **kwargs):
        self.log('G[')
        x0 = np.asfarray(x)
        #print x0
        dxs = np.zeros((len(x0), 2*len(x0)))
        for i in range(len(x0)):
            dxs[i, i] = -self.epsilon
            dxs[i, len(x0)+i] = self.epsilon
        results = [self(*(x0 + dxs[:, i], ) + args, **kwargs) for i in range(2*len(x0))]
        jac = np.zeros([len(x0), len(np.atleast_1d(results[0]))])
        for i in range(len(x0)):
            jac[i] = (results[len(x0)+i] - results[i]) / (2*self.epsilon)
        self.log(']')
        return jac.transpose()


# minimize(f, start_params = {'x': 1, 'y': [1,2,3,4.4]}, optimize = ['x'],
#          jac=None, method='SLSQP', constraints=[], args=('blub',))
# def f(x, y, *args, **kwargs)


class ParameterManager(object):
    def __init__(self, parameters, optimize, **kwargs):
        """ Create a parameter manager
            :param parameters: The parameters to manage
            :type parameters: list of strings
            :param optimize: The parameters that should be optimized. Has to be a subset of parameters
            :type optimize: list of strings
            :param **kwargs: Initial values of the parameters
        """
        self.parameters = parameters
        self.optimize = optimize
        self.param_values = kwargs

    def extract_parameters(self, x, return_list=False):
        """Return dictionary of optimization parameters from vector x.
           The non-optimization parameters will be taken from the initial values.
           if return_list==True, return a list instead of an dictionary"""
        params = self.param_values.copy()
        index = 0
        for param_name in self.optimize:
            if not isinstance(self.param_values[param_name], np.ndarray):
                # Only scalar value
                params[param_name] = x[index]
                index += 1
            else:
                shape = self.param_values[param_name].shape
                if len(shape) > 1:
                    raise ValueError('Arrays with more than one dimension are not yet supported!')
                params[param_name] = x[index:index+shape[0]]
                index += shape[0]
        if return_list:
            return [params[key] for key in self.parameters]
        else:
            return params

    def build_vector(self, **kwargs):
        """Build a vector of the optimization parameters.
           The initial values will be taken unless you overwrite
           them using the keyword arguments"""
        params = self.param_values.copy()
        params.update(kwargs)
        vector_values = [params[key] for key in self.optimize]
        return np.hstack(vector_values)

    def get_length(self, param_name):
        """Return the length of parameter param_name when it is used in the optimization vector"""
        if not isinstance(self.param_values[param_name], np.ndarray):
            # Only scalar value
            return 1
        else:
            shape = self.param_values[param_name].shape
            if len(shape) > 1:
                raise ValueError('Arrays with more than one dimension are not yet supported!')
            return shape[0]


def wrap_parameter_manager(f, parameter_manager):
    def new_f(x, *args, **kwargs):
        params = parameter_manager.extract_parameters(x, return_list = True)
        params.extend(args)
        return f(*params, **kwargs)
    return new_f


def minimize(f, parameter_manager, args=(), method='BFGS',
             jac=None,
             bounds=None,
             constraints=(),
             tol=None,
             options=None,
             jac_approx = FunctionWithApproxJacobian,
             callback=None):
    """Minimize function f with scipy.optimize.minimze, using the parameters
       and initial values from the parameter_manager.

       Remark: Notice that at least SLSQP does not support None values in the bounds"""
    wrapped_f = wrap_parameter_manager(f, parameter_manager)
    x0 = parameter_manager.build_vector()
    if callable(jac):
        def jac_with_keyword(*args, **kwargs):
            kwargs['optimize'] = parameter_manager.optimize
            ret = jac(*args, **kwargs)
            param_values = parameter_manager.param_values.copy()
            for i, param_name in enumerate(parameter_manager.optimize):
                param_values[param_name] = ret[i]
            return parameter_manager.build_vector(**param_values)
        fun_ = wrapped_f
        jac_ = wrap_parameter_manager(jac_with_keyword, parameter_manager)
    elif bool(jac):
        def func_with_keyword(*args, **kwargs):
            kwargs['optimize'] = parameter_manager.optimize
            func_value, jacs = f(*args, **kwargs)
            param_values = parameter_manager.param_values.copy()
            for i, param_name in enumerate(parameter_manager.optimize):
                param_values[param_name] = jacs[i]
            return func_value, parameter_manager.build_vector(**param_values)
        fun_ = wrap_parameter_manager(func_with_keyword, parameter_manager)
        jac_ = True
    else:
        fun = jac_approx(wrapped_f, 1e-8)
        jac_ = fun.jac
        fun_ = fun.func

    # Adapt constraints
    if isinstance(constraints, dict):
        constraints = [constraints]
    new_constraints = []
    for constraint in constraints:
        new_constraint = constraint.copy()
        new_constraint['fun'] = wrap_parameter_manager(constraint['fun'], parameter_manager)
        new_constraints.append(new_constraint)

    #Adapt bounds:
    if bounds is not None:
        new_bounds = []
        for param_name in parameter_manager.optimize:
            if param_name in bounds:
                new_bounds.extend(bounds[param_name])
            else:
                length = parameter_manager.get_length(param_name)
                for i in range(length):
                    new_bounds.append((None, None))
    else:
        new_bounds = None
    if callback is not None:
        callback = wrap_parameter_manager(callback, parameter_manager)
    res = scipy.optimize.minimize(fun_, x0, args=args, jac=jac_,
                                  method=method,
                                  constraints=new_constraints,
                                  bounds=new_bounds,
                                  tol=tol,
                                  callback=callback,
                                  options=options)

    params = parameter_manager.extract_parameters(res.x)
    for key in parameter_manager.parameters:
        setattr(res, key, params[key])
    return res

if __name__ == '__main__':
    def testfunc(x):
        return np.sum(x ** 4)
    testFun = FunctionWithApproxJacobian(testfunc, 1e-8)
    x0 = np.zeros(3)
    #val = testFun(x0)
    #print
    #print val
    g = testFun.jac(x0)
    print
    print g
