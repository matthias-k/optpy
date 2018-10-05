optpy
========

This library provides some functions to make optimization in python easier.
It can use `scipy.optimize`. Also, it provides
an interface that makes minimizing functions of multiple variables easier,
especially if only a subset of the variables should be considered for the
optimization.

This package used to contain a convenience function `minimize_ipopt` that
mimicked the `scipy.mimize.optimize` interface. This function is now part of
[https://github.com/matthias-k/cyipopt](cyipopt). If you want to use optpy
with ipopt just set the `method` argument to the minimize function from
cyipopt.


Optimization of multiple variables
----------------------------------

In practise, often one has to optimize functions that depend on multiple variables. Most
optimizers, including `scipy.optimize.minimze` and ipopt can handle only functions that
depend on one array-like parameter that should be optimized. It gets even more
complicated when one wants to include certain variables in a flexible way into
the optimimzation or keep them constant. `optpy` provides an easy way to do so:

Here we have a function of three variables `x1`, `x2`, `x3`, but we want to optimize
only `x1` and `x2`.

```python
def f(x1, x2, x3):
    return np.sum(x1**2)+np.sum(x2**2)

def constraint(x1, x2, x3):
    return x1-1.0

constraints = [{'type': 'eq', 'fun': constraint}]

res = optimization.minimize(f, {'x1': 1.0,
                                'x2': np.array([2.0, 2.0]),
                                'x3': 1.0},
                            optimize=['x1', 'x2'],
                            method='SLSQP', constraints = constraints)
np.testing.assert_allclose(res.x1, 1.0)
np.testing.assert_allclose(res.x2, [0.0, 0.0])
np.testing.assert_allclose(res.x3, 1.0)
```

There is also a more explicit interface:

```python
parameter_manager = optpy.ParameterManager(['x1', 'x2', 'x3'], ['x1', 'x2'],
                                               x1=1.0, x2=np.array([2.0, 2.0]), x3=1.0)

def f(x1, x2, x3):
    return np.sum(x1**2)+np.sum(x2**2)

def constraint(x1, x2, x3):
    return x1-1.0

constraints = [{'type': 'eq', 'fun': constraint}]

res = optpy.minimize(f, parameter_manager, method='SLSQP', constraints = constraints)
np.testing.assert_allclose(res.x1, 1.0)
np.testing.assert_allclose(res.x2, [0.0, 0.0])
np.testing.assert_allclose(res.x3, 1.0)
```

