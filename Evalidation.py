import numpy
import torch
from Aconfig import (Config, DEVICE)
from Bmodel import (PINN1, PINN2, PINN3, PINN4)
import os


config = Config()
delta = config.delta


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