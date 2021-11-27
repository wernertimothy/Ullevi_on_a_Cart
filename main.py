
# arrays
import numpy as np
# system environments
from gym import environments
# control stuff
from gym import control
# visualization
import matplotlib.pyplot as plt

# === system ===
# initialize cart pole system from gym
x0 = np.array([-0.5,np.pi,0,0]) # initial state
Ts = 0.05                        # sampletime

'''
use 'full' for the full dynamics with force input on the cart
use 'simplified' for ideal velocity control on the cart and input is the cart acceleration
'''

# equations = 'full'
equations = 'simplified'
cart_pole = environments.CartPoleSystem(x0, equations=equations)
cart_pole.samplerate = Ts

# === controller ===
# phi_bar = 0.0 # pendulum down 
phi_bar = np.pi # pendulum up
x_bar = np.array([0.5, phi_bar, 0, 0])
g_model = 9.81 # model gravity
lp_model = 0.6 # model pole length
A = np.array([
    [0,0,1,0],
    [0,0,0,1],
    [0,0,0,0],
    [0, -g_model/lp_model*np.cos(phi_bar), 0, 0]
])
B = np.array([
    [0],
    [0],
    [1],
    [np.cos(phi_bar)/lp_model]
])
Q = np.diag([1,10,0.5,1])
R = 0.5

K, P, eig = control.lqr(A, B, Q, R)

# === simulation ===
# define simulation time
t0 = 0
tf = 10
time = np.arange(t0, tf+Ts, Ts)

# trajectory allocation
X_sim = np.zeros((cart_pole.nx, len(time)))
U_sim = np.zeros(len(time))

# actual simulation
for k, t in enumerate(time):
    # measure current state
    xt = cart_pole.measure()
    # generate control input
    ut = -K@(xt-x_bar)
    if np.abs(ut) > 1.0 : ut = np.sign(ut)*1.0
    # apply control input to the system
    cart_pole.integrate(ut)
    # store simulation data
    X_sim[:,k] = xt
    U_sim[k] = ut

# visulization
cart_pole.visualize(X_sim, repeat=True)

plt.figure()
plt.plot(time, U_sim)
plt.show()