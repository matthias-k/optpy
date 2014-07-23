import unittest

import numpy as np

import optimize.ipopt_wrapper as ipopt


class TestWrapper(unittest.TestCase):
    def test_fun(self):
        fun = lambda x: np.sum(np.square(x))
        wrapper = ipopt.IpoptProblemWrapper(fun)
        x = np.array([0, 0])
        self.assertEqual(wrapper.objective(x), 0)
        np.testing.assert_allclose(wrapper.gradient(x), np.array([[0, 0]]), atol=1e-7)

    def test_fun_with_jac(self):
        fun = lambda x: np.sum(np.square(x))
        jac = lambda x: 2*x
        wrapper = ipopt.IpoptProblemWrapper(fun, jac=jac)
        x = np.array([0, 0])
        self.assertEqual(wrapper.objective(x), 0)
        np.testing.assert_allclose(wrapper.gradient(x), np.array([0, 0]))

    def test_fun_with_constraint(self):
        fun = lambda x: x
        constraint = {'fun': lambda x: x-1,
                      'type': 'ineq'}
        wrapper = ipopt.IpoptProblemWrapper(fun, constraints=constraint)
        x = np.array([0])
        np.testing.assert_allclose(wrapper.constraints(x), [-1])
        np.testing.assert_allclose(wrapper.jacobian(x), [[1]])

    def test_fun_with_constraint2d(self):
        fun = lambda x: x
        constraint = {'fun': lambda x: np.array([x[0]-1, 2*x[1]+2]),
                      'type': 'ineq'}
        wrapper = ipopt.IpoptProblemWrapper(fun, constraints=constraint)
        x = np.array([1, 1])
        np.testing.assert_allclose(wrapper.constraints(x), [0, 4])
        np.testing.assert_allclose(wrapper.jacobian(x), [[1, 0], [0, 2]])

    def test_fun_with_2constraints(self):
        fun = lambda x: x
        constraints = [{'fun': lambda x: x-1,
                        'type': 'ineq'},
                       {'fun': lambda x: -(x-10),
                        'type': 'ineq'}
                       ]
        wrapper = ipopt.IpoptProblemWrapper(fun, constraints=constraints)
        x = np.array([0])
        np.testing.assert_allclose(wrapper.constraints(x), [-1, 10])
        np.testing.assert_allclose(wrapper.jacobian(x), [[1], [-1]])

    def test_fun_with_arg(self):
        fun = lambda x, c: np.sum(np.square(x+c))
        args = (np.array([4,5]),)
        wrapper = ipopt.IpoptProblemWrapper(fun, args=args)
        x = np.array([2,3])
        self.assertEqual(wrapper.objective(x), 100)
        np.testing.assert_allclose(wrapper.gradient(x), np.array([[12, 16]]), atol=1e-7)

    def test_fun_with_constraint_with_arg(self):
        fun = lambda x: x
        constraint = {'fun': lambda x, c: x+c-1,
                      'type': 'ineq',
                      'args': ([3],)}
        wrapper = ipopt.IpoptProblemWrapper(fun, constraints=constraint)
        x = np.array([0])
        np.testing.assert_allclose(wrapper.constraints(x), [2])
        np.testing.assert_allclose(wrapper.jacobian(x), [[1]])


class TestConvertConstraints(unittest.TestCase):
    def test_eq(self):
        x0 = 0
        constraint = {'fun': lambda x: x-1,
                      'type': 'eq'}
        cl, cu = ipopt.get_constraint_bounds(constraint, x0)
        np.testing.assert_allclose(cl, [0])
        np.testing.assert_allclose(cu, [0])

    def test_eq2d(self):
        x0 = np.array([0, 0])
        constraint = {'fun': lambda x: np.asarray([x[0]-1, x[1]-2]),
                      'type': 'eq'}
        cl, cu = ipopt.get_constraint_bounds(constraint, x0)
        np.testing.assert_allclose(cl, [0, 0])
        np.testing.assert_allclose(cu, [0, 0])

    def test_ineq(self):
        x0 = 0
        inf = 1e18
        constraint = {'fun': lambda x: x-1,
                      'type': 'ineq'}
        cl, cu = ipopt.get_constraint_bounds(constraint, x0, INF=inf)
        np.testing.assert_allclose(cl, [0])
        np.testing.assert_allclose(cu, [inf])

    def test_ineq2d(self):
        x0 = np.array([0, 0])
        inf = 1e18
        constraint = {'fun': lambda x: np.asarray([x[0]-1, x[1]-2]),
                      'type': 'ineq'}
        cl, cu = ipopt.get_constraint_bounds(constraint, x0,INF=inf)
        np.testing.assert_allclose(cl, [0, 0])
        np.testing.assert_allclose(cu, [inf, inf])

    def test_eq_and_ineq(self):
        x0 = 0
        inf = 1e18
        constraints = [{'fun': lambda x: x-1,
                        'type': 'eq'},
                       {'fun': lambda x: -(x-10),
                        'type': 'ineq'}]
        cl, cu = ipopt.get_constraint_bounds(constraints, x0, INF=inf)
        np.testing.assert_allclose(cl, [0, 0])
        np.testing.assert_allclose(cu, [0, inf])

    def test_eq2d_and_ineq(self):
        x0 = 0
        inf = 1e18
        constraints = [{'fun': lambda x: np.array([x-1, 2*x-2]),
                        'type': 'eq'},
                       {'fun': lambda x: -(x-10),
                        'type': 'ineq'}]
        cl, cu = ipopt.get_constraint_bounds(constraints, x0, INF=inf)
        np.testing.assert_allclose(cl, [0, 0, 0])
        np.testing.assert_allclose(cu, [0, 0, inf])

class TestMinimize(unittest.TestCase):
    def test_fun1d(self):
        fun = lambda x: x**2
        res = ipopt.minimize_ipopt(fun, 1000)
        self.assertIsInstance(res.x, float)
        np.testing.assert_allclose(res.x, 0, atol=1e-8)
        self.assertTrue(res.success)
        self.assertEqual(res.status, 0)
        np.testing.assert_allclose(res.fun, 0, atol=1e-8)

    def test_fun2d(self):
        fun = lambda x: np.sum(np.square(x))
        res = ipopt.minimize_ipopt(fun, [1000, 1], tol=1e-9)
        self.assertIsInstance(res.x, np.ndarray)
        np.testing.assert_allclose(res.x, [0, 0], atol=1e-8)
        self.assertTrue(res.success)
        self.assertEqual(res.status, 0)
        np.testing.assert_allclose(res.fun, 0, atol=1e-8)

    def test_fun1d_with_constraint1d(self):
        fun = lambda x: x**2
        constraints = {'fun': lambda x: x-1,
                       'type': 'ineq'}
        res = ipopt.minimize_ipopt(fun, 1000, constraints=constraints, tol=1e-9)
        self.assertIsInstance(res.x, float)
        np.testing.assert_allclose(res.x, 1, atol=1e-8)
        self.assertTrue(res.success)
        self.assertEqual(res.status, 0)
        np.testing.assert_allclose(res.fun, 1, atol=1e-8)

    def test_fun2d_with_constraint_1d(self):
        fun = lambda x: np.sum(np.square(x))
        constraints = {'fun': lambda x: x[0]-1,
                       'type': 'ineq'}
        res = ipopt.minimize_ipopt(fun, [1000, 10], constraints=constraints, tol=1e-9)
        self.assertIsInstance(res.x, np.ndarray)
        np.testing.assert_allclose(res.x, [1, 0], atol=1e-8)
        self.assertTrue(res.success)
        self.assertEqual(res.status, 0)
        np.testing.assert_allclose(res.fun, 1, atol=1e-8)

    def test_fun2d_with_constraint_2d(self):
        fun = lambda x: np.sum(np.square(x))
        constraints = {'fun': lambda x: np.array([x[0]-1, x[1]-2]),
                       'type': 'ineq'}
        res = ipopt.minimize_ipopt(fun, [1000, 10], constraints=constraints, tol=1e-9, options={'disp': 0})
        self.assertIsInstance(res.x, np.ndarray)
        np.testing.assert_allclose(res.x, [1, 2], atol=1e-8)
        self.assertTrue(res.success)
        self.assertEqual(res.status, 0)
        np.testing.assert_allclose(res.fun, 5, atol=1e-8)

    def test_fun2d_with_2constraints(self):
        fun = lambda x: np.sum(np.square(x))
        constraints = [{'fun': lambda x: x[0]-1,
                        'type': 'ineq'},
                        {'fun': lambda x: x[1]-2,
                        'type': 'ineq'}]

        res = ipopt.minimize_ipopt(fun, [1000, 10], constraints=constraints, tol=1e-9, options={'disp': 0})
        self.assertIsInstance(res.x, np.ndarray)
        np.testing.assert_allclose(res.x, [1, 2], atol=1e-8)
        self.assertTrue(res.success)
        self.assertEqual(res.status, 0)
        np.testing.assert_allclose(res.fun, 5, atol=1e-8)

if __name__ == '__main__':
    unittest.main()
