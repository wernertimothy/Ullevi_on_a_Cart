
# arrays
import numpy as np
# system environments
from gym import environments

# initialize cart pole system from gym
x0 = np.array([0,np.pi/4,0,0]) # initial state
Ts = 0.05                # sampletime

cart_pole = environments.CartPoleSystem(x0)
cart_pole.samplerate = Ts

# define simulation time
t0 = 0
tf = 10
time = np.arange(t0, tf+Ts, Ts)

# trajectory allocation
X_sim = np.zeros((cart_pole.nx, len(time)))
U_sim = np.zeros(len(time))

for k, t in enumerate(time):
    # measure current state
    xt = cart_pole.measure()
    # generate control input
    ut = 0
    # apply control input to the system
    cart_pole.integrate(ut)
    # store simulation data
    X_sim[:,k] = xt
    U_sim[k] = ut

cart_pole.visualize(X_sim)