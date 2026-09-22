from Aconfig import Config
from Aconfig import DEVICE
import torch
import numpy as np
import matplotlib.pyplot as plt

config = Config()
delta = config.delta


x_min = config.x_min
x_max = config.x_max
t_min = config.t_min
t_max = config.t_max

x_min.to(DEVICE), x_max.to(DEVICE), t_min.to(DEVICE), t_max.to(DEVICE)

t_mid = (t_max + t_min)/2
x_mid = (x_max + x_min)/2

def subdomain1(delta):

    x1_min = x_min
    x1_max = x_mid - delta

    t1_min = t_min
    t1_max = t_mid - delta

    return x1_min, x1_max, t1_min, t1_max


def subdomain2(delta):
    x2_min = x_mid + delta
    x2_max = x_max
    t2_min = t_min
    t2_max = t_mid - delta

    return x2_min, x2_max, t2_min, t2_max


def subdomain3(delta):
    x3_min = x_min
    x3_max = x_mid - delta
    t3_min = t_mid + delta
    t3_max = t_max

    return x3_min, x3_max, t3_min, t3_max


def subdomain4(delta):
    x4_min = x_mid + delta
    x4_max = x_max
    t4_min = t_mid + delta
    t4_max = t_max

    return x4_min, x4_max, t4_min, t4_max


def subdomain5(delta, N=2000):

    # Generate candidate points
    x = torch.rand(N * 10, 1)
    t = torch.rand(N * 10, 1)

    # Points in the four corner subdomains
    mask = ~(
        (
            (x <= x_mid - delta) &
            (t <= t_mid - delta)
        )
        |
        (
            (x >= x_mid + delta) &
            (t <= t_mid - delta)
        )
        |
        (
            (x <= x_mid - delta) &
            (t >= t_mid + delta)
        )
        |
        (
            (x >= x_mid + delta) &
            (t >= t_mid + delta)
        )
    )

    x5 = x[mask]
    t5 = t[mask]

    return x5[:N], t5[:N]



# x5, t5 = subdomain5(delta, N=2000)

# print(x5.shape)
# print(t5.shape)




# N = 2000

# x1_min, x1_max, t1_min, t1_max = subdomain1(delta)

# x1 = x1_min + (x1_max - x1_min) * torch.rand(N,1)
# t1 = t1_min + (t1_max - t1_min) * torch.rand(N,1)
# X = torch.cat((x1,t1), dim=1)


# print(x1.shape)
# print(t1.shape)
# print(X.shape)


