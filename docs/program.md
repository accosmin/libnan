# Linear and quadratic programming module


#### Introduction

Libnano provides various methods to solve linear and quadratic programs. The goal is to find the minimum `x` of either a linear or a quadratic combination of its components, contrained optionally with affine equalities and inequalities. The returned point `x` is guaranteed to be the global minimum if the program is feasible. More details can be found in the following sources:


* "Primal-dual interior-point methods", S. Wright, 1997
* "Convex Optimization", S. Boyd, L. Vandenberghe, 2004
* "Numerical Optimization", J. Nocedal, S. Wright, 2006


#### Program definition

Linear and quadratic programs can be defined using the associated classes [linear_program_t](../include/nano/function/linear.h) and [quadratic_program_t](../include/nano/function/quadratic.h). Optionally arbitrarily many linear equality and inequality constraints can be chained by calling either the `make_linear` or the `make_quadratic` utilities. Following we show how to define various programs using the C++ interface:

* the `standard linear program` consists of equality contraints and positive element-wise solutions:
```
 min  c.dot(x)                          const auto program = make_linear(c,
 s.t. A * x = b,                              make_equality(A, b),
      x >= 0.                                 make_greater(c.size(), 0.0));
```

* the `inequality linear program` consists of only inequality constraints:
```
 min  c.dot(x)                          const auto program = make_linear(c,
 s.t. A * x <= b.                             make_inequality(A, b));
```

* the `general linear program` consists of both equality and inequality constraints:
```
 min  c.dot(x)                          const auto program = make_linear(c,
 s.t. A * x = b,                              make_equality(A, b),
      G * x <= h.                             make_inequality(G, h));
```

* the `rectangle linear program` consists of element-wise inequality constraints:
```
 min  c.dot(x)                          const auto program = make_linear(c,
 s.t. A * x = b,                              make_equality(A, b),
      l <= x <= u.                            make_greater(l), make_less(u));
```

* the `general quadratic program` consists of both equality and inequality constraints:
```
 min  1/2 x.dot(Q * x) + c.dot(x)       const auto program = make_quadratic(Q, c,
 s.t. A * x = b,                              make_equality(A, b),
      G * x <= h.                             make_inequality(G, h));
```
Note that all equalities and inequalities above are specified element-wise.


More specifically the following example code extracted from the [linear programming example](../example/src/linprog.cpp):
```
const auto n_equals = 2;
const auto c        = make_vector<scalar_t>(1, 1, 1);
const auto A        = make_matrix<scalar_t>(n_equals, 2, 1, 0, 1, 0, 1);
const auto b        = make_vector<scalar_t>(4, 1);

const auto program  = make_linear(c, make_equality(A, b), make_greater(c.size(), 0.0));
```
illustrates how to define the following standard-form linear program:
```
 min  x1 + x2 + x3
 s.t. 2 * x1 + x2 = 4,
      x1 + x3 = 1,
      (x1, x2, x3) >= 0.
```
with the obvious solution `(1, 2, 0)`.


#### Solution

The linear and the quadratic programs are solved typically with variations of either the simplex method or the `primal-dual interior point method`. Libnano implements the latter as described in the references above. This variation is relatively easy to code and understand and it can be easily tuned for specific applications.

The following example code extracted from the [linear programming example](../example/src/linprog.cpp) shows how to solve the linear program defined above:
```
auto solver                           = solver_t{};
solver.parameter("solver::epsilon")   = 1e-12;
solver.parameter("solver::max_iters") = 100;
solver.logger(make_stdout_logger());

const auto program = make_linear(c, make_equality(A, b), make_greater(c.size(), 0));
const auto state   = solver.solve(program);

std::cout << std::fixed << std::setprecision(12)
          << "solution: x=" << state.m_x.transpose() << std::endl;

assert(state.m_status == solver_status::converged);
assert(close(state.m_x, xbest, 1e-10));
```

Please refer to the [quadratic programming example](../example/src/quadprog.cpp) for a similar example on how to solve quadratic programs.


#### Hints

* Typically the solution is found with 8-10 decimals in less than 20 iterations with the default settings. If convergence is not achieved, then the program is most likely unfeasible or unbounded.

* Convexity can be checked for quadratic problems using `program.convex()` function.

* If the number of equality constraints is larger than the number of variables or the constraints may be linear dependent, then it is better to call `program.reduce()` to transform the equality constraints `A` in a full row-rank matrix. This speeds-up the solver and often improves accuracy.
