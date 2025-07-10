# import packages
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.distributions.binomial import Binomial

from dataclasses import dataclass

# Simple Deterministic SI Model
class si_model_det():
    def __init__(self, beta):
        self.N = 100
        self.I_init = 20
        self.S_init = self.N - self.I_init
        self.beta = beta

        self.S = self.S_init
        self.I = self.I_init

        self.transition_history = []

    def forward(self, num_step):
        for idx_step in range(num_step):
            transition_S2I = self.S * self.beta * self.I / self.N
            # update
            self.S -= transition_S2I
            self.I += transition_S2I

            self.transition_history.append(transition_S2I)
        return self.transition_history


si1 = si_model_det(0.25)
true_transition_history = np.round(si1.forward(10))
true_transition_history = torch.tensor(true_transition_history, dtype=torch.float32)

# Changed from Sonny's code

# Note: F.gumbel_softmax() does the sampling
# It does not take a Generator instance -- instead, it relies on the global RNG state
# If we wanted to be more careful about random number generation, we would want to code
#     the gumbel softmax function ourselves and make it a function of a Generator instance
torch.manual_seed(888888)

@dataclass
class State:
    I: torch.tensor
    S: torch.tensor


@dataclass
class Params:
    N: torch.tensor
    beta: torch.tensor


def get_transition(state: State, params: Params):
    # probability of success
    probs = params.beta * state.I / params.N

    # compute log PMF
    # create Binomial distribution object
    binomial = Binomial(state.S, probs)

    # and define the values at which to evaluate the PMF
    successes = torch.arange(state.S.item() + 1)

    # compute log PMF
    log_pmf = binomial.log_prob(successes)

    # Gumbel-Softmax
    samples = F.gumbel_softmax(log_pmf, tau=1, hard=True)

    # compute transitions
    transition = torch.sum(samples * successes)

    return transition


def simulate(state: State, params: Params, num_timesteps: int):

    transition_history = []

    for t in range(num_timesteps):
        transition = get_transition(state, params)
        state = State(S = state.S - transition, I = state.I + transition)
        transition_history.append(transition.clone())

    return torch.stack(transition_history)


# Manual optimization

manual_opt_beta_history = []

opt_state = State(I=torch.tensor(20.0, requires_grad=True), S=torch.tensor(80.0, requires_grad=True))
opt_params = Params(N=torch.tensor(100.0, requires_grad=True), beta=torch.tensor(0.9, requires_grad=True))

num_samples = 50

for i in range(200):

    transition_history_samples = []

    for sample in range(num_samples):

        transition_history = []

        opt_state = State(I=torch.tensor(20.0, requires_grad=True), S=torch.tensor(80.0, requires_grad=True))
        opt_params = Params(N=torch.tensor(100.0, requires_grad=True), beta=opt_params.beta)

        transition_history.append(simulate(opt_state, opt_params, 10))

    loss = F.mse_loss(torch.stack(transition_history).mean(dim=0), true_transition_history)

    loss.backward()

    # Sonny's code here
    # Disable the autograd
    with torch.no_grad():

        # inplace updates
        opt_params.beta.sub_(0.01 * opt_params.beta.grad / (1 + i))
        # clamp
        opt_params.beta.clamp_(min=0.001, max=0.999)

        # if sample == num_samples - 1:
        #     print("Iteration {}, loss: {:.4f}, grad: {:.4f}, new beta: {:.4f}".format(i + 1, loss.item(),
        #                                                                               opt_params.beta.grad.item(),
        #                                                                               opt_params.beta.item()))
        manual_opt_beta_history.append(opt_params.beta.item())

        # zero gradients
        opt_params.beta.grad.zero_()

#### Adam optimization

opt_state = State(I=torch.tensor(20.0, requires_grad=True), S=torch.tensor(80.0, requires_grad=True))
opt_params = Params(N=torch.tensor(100.0, requires_grad=True), beta=torch.tensor(0.9, requires_grad=True))

adam_opt_beta_history = []

optimizer = torch.optim.Adam([opt_params.beta], lr=0.01)

for i in range(200):

    transition_history_samples = []

    for sample in range(num_samples):

        transition_history = []

        opt_state = State(I=torch.tensor(20.0, requires_grad=True), S=torch.tensor(80.0, requires_grad=True))
        opt_params = Params(N=torch.tensor(100.0, requires_grad=True), beta=opt_params.beta)

        transition_history.append(simulate(opt_state, opt_params, 10))

    loss = F.mse_loss(torch.stack(transition_history).mean(dim=0), true_transition_history)

    print(loss)

    loss.backward()

    optimizer.step()

    adam_opt_beta_history.append(opt_params.beta.item())

    optimizer.zero_grad()

# Plotting

plt.plot(adam_opt_beta_history, label="Adam beta")
plt.plot(manual_opt_beta_history, label="Manual optimization beta")
plt.axhline(y=0.25, color="k", linestyle=":", label="True beta")
plt.legend()
plt.show()

breakpoint()