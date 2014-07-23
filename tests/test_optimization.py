import numpy as np
import unittest

from optimize import optimization


class TestFunctionWithApproxJacobian(unittest.TestCase):
    def test_function(self):
        def f(x):
            return np.sum(x**2)
        func = optimization.FunctionWithApproxJacobian(f, epsilon=1e-8)
        point = np.array([1.0, 2.0, 3.0])
        self.assertEqual(func(point), 14.0)
        np.testing.assert_allclose(func.jac(point), [[2.0, 4.0, 6.0]])


class TestParameterManager(unittest.TestCase):
    def test_build_vector(self):
        parameter_manager = optimization.ParameterManager(['x1', 'x2', 'x3'], ['x1', 'x2'],
                                                          x1=1.0, x2=np.array([2.0, 3.0]), x3=4.0)
        np.testing.assert_allclose(parameter_manager.build_vector(), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(parameter_manager.build_vector(x1=4), [4.0, 2.0, 3.0])

    def test_extract_parameters(self):
        parameter_manager = optimization.ParameterManager(['x1', 'x2', 'x3'], ['x1', 'x2'],
                                                          x1=1.0, x2=np.array([2.0, 3.0]), x3=4.0)
        x = np.array([5.0, 6.0, 7.0])
        params = parameter_manager.extract_parameters(x)
        np.testing.assert_allclose(params['x1'], 5.0)
        np.testing.assert_allclose(params['x2'], [6.0, 7.0])
        np.testing.assert_allclose(params['x3'], 4.0)


class TestMinimize(unittest.TestCase):
    def test_simple(self):
        parameter_manager = optimization.ParameterManager(['x1', 'x2', 'x3'], ['x1', 'x2'],
                                                          x1=1.0, x2=np.array([2.0, 2.0]), x3=1.0)

        def f(x1, x2, x3):
            return np.sum(x1**2)+np.sum(x2**2)

        res = optimization.minimize(f, parameter_manager, method='SLSQP')
        np.testing.assert_allclose(res.x1, 0.0)
        np.testing.assert_allclose(res.x2, [0.0, 0.0])
        np.testing.assert_allclose(res.x3, 1.0)

    def test_with_equality_constraint(self):
        parameter_manager = optimization.ParameterManager(['x1', 'x2', 'x3'], ['x1', 'x2'],
                                                          x1=1.0, x2=np.array([2.0, 2.0]), x3=1.0)

        def f(x1, x2, x3):
            return np.sum(x1**2)+np.sum(x2**2)

        def constraint(x1, x2, x3):
            return x1-1.0

        constraints = [{'type': 'eq', 'fun': constraint}]

        res = optimization.minimize(f, parameter_manager, method='SLSQP', constraints = constraints)
        np.testing.assert_allclose(res.x1, 1.0)
        np.testing.assert_allclose(res.x2, [0.0, 0.0])
        np.testing.assert_allclose(res.x3, 1.0)

    def test_with_inequality_constraint(self):
        parameter_manager = optimization.ParameterManager(['x1', 'x2', 'x3'], ['x1', 'x2'],
                                                          x1=1.0, x2=np.array([2.0, 0.0]), x3=1.0)

        def f(x1, x2, x3):
            return np.sum(x1**2)+np.sum(x2**2)

        def constraint(x1, x2, x3):
            return x2.sum()-1

        constraints = [{'type': 'ineq', 'fun': constraint}]

        res = optimization.minimize(f, parameter_manager, method='SLSQP', constraints = constraints)
        np.testing.assert_allclose(res.x1, 0.0, atol=1e-8)
        np.testing.assert_allclose(res.x2, [0.5, 0.5])
        np.testing.assert_allclose(res.x3, 1.0)

    def test_bounds(self):
        parameter_manager = optimization.ParameterManager(['x1', 'x2', 'x3'], ['x1', 'x2'],
                                                          x1=1.0, x2=np.array([2.0, 2.0]), x3=1.0)

        def f(x1, x2, x3):
            return np.sum(x1**2)+np.sum(x2**2)

        res = optimization.minimize(f, parameter_manager, method='SLSQP',
                                    bounds={'x1': [(0.5, 3.0)], 'x2': [(-2, 4), (-2, 10)]}, tol=1e-11)
        np.testing.assert_allclose(res.x1, 0.5)
        np.testing.assert_allclose(res.x2, [0.0, 0.0], atol=1e-8)
        np.testing.assert_allclose(res.x3, 1.0)

    def partial_bounds(self):
        parameter_manager = optimization.ParameterManager(['x1', 'x2', 'x3'], ['x1', 'x2'],
                                                          x1=1.0, x2=np.array([2.0, 2.0]), x3=1.0)

        def f(x1, x2, x3):
            return np.sum(x1**2)+np.sum(x2**2)

        res = optimization.minimize(f, parameter_manager, method='SLSQP',
                                    bounds={'x2': [(-2, 4), (1, 10)]}, tol=1e-11)
        np.testing.assert_allclose(res.x1, 0.0)
        np.testing.assert_allclose(res.x2, [0.0, 1.0], atol=1e-8)
        np.testing.assert_allclose(res.x3, 1.0)

    def test_args(self):
        parameter_manager = optimization.ParameterManager(['x1', 'x2', 'x3'], ['x1', 'x2'],
                                                          x1=1.0, x2=np.array([2.0, 2.0]), x3=1.0)

        def f(x1, x2, x3, argument):
            self.assertEqual(argument, 'argument')
            return np.sum(x1**2)+np.sum(x2**2)

        res = optimization.minimize(f, parameter_manager, args= ('argument', ), method='SLSQP')
        np.testing.assert_allclose(res.x1, 0.0)
        np.testing.assert_allclose(res.x2, [0.0, 0.0])
        np.testing.assert_allclose(res.x3, 1.0)

    def test_separate_jacobian(self):
        parameter_manager = optimization.ParameterManager(['x1', 'x2', 'x3'], ['x1', 'x2'],
                                                          x1=1.0, x2=np.array([2.0, 2.0]), x3=1.0)

        def f(x1, x2, x3):
            return np.sum(x1**2)+np.sum(x2**2)

        def fprime(x1, x2, x3, optimize=None):
            ret = []
            if 'x1' in optimize:
                ret.append(2*x1)
            if 'x2' in optimize:
                ret.append(2*x2)
            if 'x3' in optimize:
                ret.append(np.zeros_like(x3))
            return ret

        res = optimization.minimize(f, parameter_manager, jac = fprime, method='SLSQP')
        np.testing.assert_allclose(res.x1, 0.0)
        np.testing.assert_allclose(res.x2, [0.0, 0.0])
        np.testing.assert_allclose(res.x3, 1.0)

    def test_combined_jacobian(self):
        parameter_manager = optimization.ParameterManager(['x1', 'x2', 'x3'], ['x1', 'x2'],
                                                          x1=1.0, x2=np.array([2.0, 2.0]), x3=1.0)

        def f(x1, x2, x3, optimize=None):
            val = np.sum(x1**2)+np.sum(x2**2)
            jacs = []
            if 'x1' in optimize:
                jacs.append(2*x1)
            if 'x2' in optimize:
                jacs.append(2*x2)
            if 'x3' in optimize:
                jacs.append(np.zeros_like(x3))
            return val, jacs

        res = optimization.minimize(f, parameter_manager, jac = True, method='SLSQP')
        np.testing.assert_allclose(res.x1, 0.0)
        np.testing.assert_allclose(res.x2, [0.0, 0.0])
        np.testing.assert_allclose(res.x3, 1.0)



if __name__ == '__main__':
    unittest.main()
