# Tests README

## Important notes

### pytest fixtures are AMAZING!
- For those unfamiliar please read more -- for example, [here](https://docs.pytest.org/en/6.2.x/fixture.html)
- Fixtures are recreated for each test, which means that each test 
gets a fresh instance and we do not have to rest the simulation or
state between tests.

### Warning: Poisson and Binomial Taylor Approximation 
- In summary, Poisson and Binomial Taylor Approximation can cause
tests to fail even if there is no implementation error! These failures
are caused by poorly chosen parameter values and discretization error.
- We are currently excluding Poisson transition types from automatic test 
run. Poisson transition types are not always well-behaved -- this is
because the Poisson distribution is unbounded. For example, values 
of `beta_baseline` that are too high can result in negative compartment 
populations. Similarly, if `num_timesteps` is small, this can result 
in negative compartment populations. Sometimes tests fail because 
these choices of parameter values and parameter initial values are 
unsuitable for well-behaved Poisson random variables.
- If there are too few `timesteps_per_day` (in each `SubpopModel`'s
`SimulationSettings`), then it is possible for Binomial Taylor Approximation
(both stochastic and deterministic) to fail. This is because
the input into the "probability parameter" for the numpy
random binomial draw may be not be in [0,1]. Thus, the
Binomial Taylor Approximation transition type may not reliably
pass all tests with arbitrary simulation settings.