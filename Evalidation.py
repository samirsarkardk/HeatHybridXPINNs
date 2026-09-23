import numpy as np
import torch
from Aconfig import (Config, DEVICE)
from Bmodel import (PINN1, PINN2, PINN3, PINN4)
import os
from Dmain import CombinedModel
import matplotlib.pyplot as plt

N = 2000

config = Config()
delta = config.delta

model1 = PINN1().to(DEVICE)
model2 = PINN2().to(DEVICE)
model3 = PINN3().to(DEVICE)
model4 = PINN4().to(DEVICE)
model5 = CombinedModel(model1, model2, model3, model4).to(DEVICE)


model1.load_state_dict(torch.load("model1.pth", map_location=DEVICE))
model2.load_state_dict(torch.load("model2.pth", map_location=DEVICE))
model3.load_state_dict(torch.load("model3.pth", map_location=DEVICE))
model4.load_state_dict(torch.load("model4.pth", map_location=DEVICE))

model1.eval()
model2.eval()
model3.eval()
model4.eval()
model5.eval()

x_min = config.x_min
x_max = config.x_max
t_min = config.t_min
t_max = config.t_max
x_min.to(DEVICE), x_max.to(DEVICE), t_min.to(DEVICE), t_max.to(DEVICE)

t_mid = (t_max + t_min)/2
x_mid = (x_max + x_min)/2

x = x_min + (x_max - x_min) * torch.rand(N,1)
t = t_min + (t_max - t_min) * torch.rand(N,1)

x_mid.to(DEVICE), t_mid.to(DEVICE), x.to(DEVICE), t.to(DEVICE)

def final_solution(x, t, model1, model2, model3, model4, model5,
                   x_mid, t_mid, delta):

    x = x.reshape(-1, 1).to(DEVICE)
    t = t.reshape(-1, 1).to(DEVICE)

    # Four corner subdomains
    mask1 = (x <= x_mid - delta) & (t <= t_mid - delta)
    mask2 = (x >= x_mid + delta) & (t <= t_mid - delta)
    mask3 = (x <= x_mid - delta) & (t >= t_mid + delta)
    mask4 = (x >= x_mid + delta) & (t >= t_mid + delta)

    # Central cross-shaped subdomain
    mask5 = ~(mask1 | mask2 | mask3 | mask4)

    with torch.no_grad():
        u1 = model1(x, t)
        u2 = model2(x, t)
        u3 = model3(x, t)
        u4 = model4(x, t)

        # u5 = 0.25 * (u1 + u2 + u3 + u4)
        u5 = model5(x, t)

    u_final = (
        u1 * mask1.float()
        + u2 * mask2.float()
        + u3 * mask3.float()
        + u4 * mask4.float()
        + u5 * mask5.float()
    )

    return u_final


def exact_solution(x, t):
    return torch.exp(-torch.pi**2 * t) * torch.sin(torch.pi * x)


u_pred = final_solution(
    x, t, model1, model2, model3, model4,model5,x_mid, t_mid, delta)

u_exact = exact_solution(x, t)

relative_l2_error = (
    torch.linalg.norm(u_pred - u_exact)
    / torch.linalg.norm(u_exact)
)

print(f"Relative L2 error: {relative_l2_error.item():.6e}")
print(f"Relative L2 error: {100 * relative_l2_error.item():.4f}%")

np.save("predicted_solution.npy", u_pred.detach().cpu().numpy())


Nx, Nt = 200, 200

x_values = np.linspace(x_min.item(), x_max.item(), Nx)
t_values = np.linspace(t_min.item(), t_max.item(), Nt)
X, T = np.meshgrid(x_values, t_values)

x_grid = torch.tensor(X.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
t_grid = torch.tensor(T.reshape(-1, 1), dtype=torch.float32, device=DEVICE)

u_pred_grid = final_solution(
    x_grid, t_grid, model1, model2, model3, model4,model5,
    x_mid, t_mid, delta
).cpu().numpy().reshape(Nt, Nx)

u_exact_grid = np.exp(-np.pi**2 * T) * np.sin(np.pi * X)
absolute_error = np.abs(u_pred_grid - u_exact_grid)

fig, ax = plt.subplots(1, 3, figsize=(18, 5))

vmin = min(u_exact_grid.min(), u_pred_grid.min())
vmax = max(u_exact_grid.max(), u_pred_grid.max())

im0 = ax[0].pcolormesh(X, T, u_exact_grid, shading="auto",
                       cmap="viridis", vmin=vmin, vmax=vmax)
ax[0].set_title("Exact solution")
ax[0].set_xlabel("x")
ax[0].set_ylabel("t")
fig.colorbar(im0, ax=ax[0])

im1 = ax[1].pcolormesh(X, T, u_pred_grid, shading="auto",
                       cmap="viridis", vmin=vmin, vmax=vmax)
ax[1].set_title("Predicted solution")
ax[1].set_xlabel("x")
ax[1].set_ylabel("t")
fig.colorbar(im1, ax=ax[1])

im2 = ax[2].pcolormesh(X, T, absolute_error, shading="auto", cmap="magma")
ax[2].set_title("Absolute error")
ax[2].set_xlabel("x")
ax[2].set_ylabel("t")
fig.colorbar(im2, ax=ax[2])

plt.tight_layout()
plt.savefig("solution_comparison.png", dpi=300)
plt.show()