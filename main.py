
# arrays
import numpy as np
# system environments
from gym import environments
# visualization
import matplotlib.pyplot as plt


x0 = np.array([0,0,0,0])
cart_pole = environments.CartPoleSystem(x0)


Ts = 0.05
cart_pole.samplerate = Ts

t0 = 0
tf = 10
time = np.arange(t0, tf+Ts, Ts)

env_figure = 4
plt.figure(env_figure)

for t in time:
    u = 0.1
    cart_pole.draw(env_figure)
    cart_pole.integrate(u)
