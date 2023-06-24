# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import torch
from torch import nn, optim
from matplotlib import pyplot as plt

# %%
N = 7; σ = 0.1

torch.manual_seed(0)
# x = torch.rand(N) * 2 - 1
x = torch.linspace(-1, 1, N)
y = .3 * x + torch.randn(N) * σ
p = torch.randperm(N)
x, y = x[p], y[p]


# %%
def plot_data():
    plt.scatter(x, y, facecolor='none', edgecolors='w', s=80, label='data', zorder=20)
    plt.axis('equal')
    plt.grid('on')
    plt.xlim((-1.5, 1.5))


# %%
def fit_data(D=10):
    # Define encoder
    encoder = nn.Sequential(
        nn.Linear(1, D),
        nn.ReLU(),
    )
    with torch.no_grad():
        h = encoder(x[:,None])  # N x D
    # Decoder weights, 1 x D
    if D < 100:
        w = y[None] @ h @ torch.inverse(h.t() @ h + 1e-5 * torch.eye(D))
    else:
        w = y[None] @ h @ torch.inverse(h.t() @ h + 1e-2 * torch.eye(D))
    # Generate predictions
    X = torch.linspace(-1.5, 1.5, 1000)
    with torch.no_grad():
        H = encoder(X[:,None])
    Ỹ = H @ w.t()
    # Plot
    plt.plot(X, Ỹ, label=f'{D = }')
    plt.ylim((-1, 1))


# %%
plt.style.use(['dark_background', 'bmh'])
plt.rc('axes', facecolor='k')
plt.rc('figure', facecolor='k')

# %%
torch.manual_seed(1)
plt.figure(dpi=300)
plot_data()
for e in (.5, 1.5, 3.5):
    D = int(10**e)
    fit_data(D)
plt.legend()

# %%
