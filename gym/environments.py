# typehinting
from typing import Callable, List
# arrays
import numpy as np
# visualization
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches


class CartPoleSystem:
    _equation_forms = ['simplified', 'full']

    def __init__(self, initial_condition : np.ndarray, equations : str = 'full') -> None:
        '''
        Implemenation of inverted pendulum on a cart.
    
                       O mp
                  lp  /
              ______ /__
             |      /   | mc
        f -->|     /    |
             |____|_____|
                  |      
                  |
                  |<--phi = 0
                  O
         |-----> s
    
        The state is position of the cart s, angle of the pendulum phi
        the velocity of the cart ds and the angular velocity dphi.
        The input is the force f acting on the cart.
    
               x:= [ s, phi, ds, dphi ]^\top
               u:= f
        '''
        # === system dynamic realted properties ===
        self._parameter = np.array([
            0.6,      # lp: length of the pole
            0.1,      # mp: mass of the pole
            0.5,      # mc: mass of the cart
            9.81,     # g: acceleration due to gravity
            0.4,      # dc: damping on the cart
            0.3,      # dp: damping on the pole  
        ])
        self._nx = 4 # state dim
        self._nu = 1 # input dim
        self._samplerate = 0.01
        self._state = initial_condition
        if equations in self._equation_forms:
            self._equation_form = equations
        else:
            raise Exception(f'equations can only be {*self._equation_forms,}')

        # === numeric integration related properties ===
        self._order = 4
        self._A = np.array([
            [0, 0, 0, 0],
            [1 / 2, 0, 0, 0],
            [0, 1 / 2, 0, 0],
            [0, 0, 1, 0]
        ])
        self._B = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
        self._C = np.array([0, 1 / 2, 1 / 2, 1])
        self._K = np.zeros((self._order, self._nx))

        # === visualization related properties
        self._CART_HIGHT = 0.1
        self._CART_WIDTH = 0.3
        self._CART_COLOR = np.array([245, 103, 196])/255
        self._POLE_COLOR = np.array([37, 194, 51])/255

    @property
    def nx(self) -> int:
        '''
        state dimension.
        '''
        return self._nx

    @property
    def nu(self) -> int:
        '''
        input dimension.
        '''
        return self._nu

    @property
    def samplerate(self) -> float:
        return self._samplerate

    @samplerate.setter
    def samplerate(self, samplerate : float):
        self._samplerate = samplerate
        
    def _f(self, x:np.ndarray, u:float) -> np.ndarray:
        '''
        Evaluates the vectorfield.
        see: https://underactuated.mit.edu/acrobot.html#cart_pole
        '''
        # read state
        s, phi, ds, dphi = x
        # read parameter
        lp, mp, mc, g, dc, dp = self._parameter
        # precalculation
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        # evaluate vectorfield
        if self._equation_form == 'simplified':
            dds = u
            ddphi = -g/lp*sin_phi - dp/(mp*lp*2)*dphi + 1/lp*cos_phi*u
        elif self._equation_form == 'full':
            tmp = mc + mp*sin_phi**2
            # derivatives
            dds = 1/tmp*(mp*sin_phi*(lp*dphi**2+g*cos_phi)-dc*ds+u)
            ddphi = 1/(lp*tmp)*(-mp*lp*dphi**2*cos_phi*sin_phi-(mc+mp)*g*sin_phi-dp*dphi-cos_phi*u)
        else:
            raise Exception(f'equations can only be {*self._equation_forms,}')

        return np.array([ds, dphi, dds, ddphi])


    def _step(self, f: Callable, x: np.ndarray, u: float) -> np.ndarray:
        '''
        Performs a Runge-Kutta integration step.
        '''
        h, A, B, C, K = self._samplerate, self._A, self._B, self._C, self._K
        K[0] = f(x, u)
        for s, (a, c) in enumerate(zip(A[1:], C[1:]), start=1):
            dx = np.dot(K.T, a)
            K[s] = f(x + dx * h, u)
        x_next = x + h * np.dot(B, K)
        return x_next

    def integrate(self, u : float) -> None:
        '''
        Integrates the system for one timestep.
        '''
        self._state = self._step(self._f, self._state, u)

    def measure(self) -> np.ndarray:
        '''
        Returns the current measurement.
        '''
        return self._state

    def _animate_system(self, i, X:np.ndarray) -> List:
        '''
        Defines a frame in the animation.
        '''
        # get states
        s, phi, _, _ = X[:,i]
        # get parameter
        lp, _, _, _, _, _ = self._parameter

        # draw cart
        self._cart_patch.set_xy([s-self._CART_WIDTH/2, 0-self._CART_HIGHT/2])
        # draw pole
        x_pole = (s, s + np.sin(phi)*lp)
        y_pole = (0, -np.cos(phi)*lp)
        self._pole_patch.set_data(x_pole, y_pole)

        return self._fig_geoms

    def visualize(self, state_trjectory : np.ndarray, repeat = True) -> None:
        '''
        Visualizes a given state trajectory.
        '''
        # set figure
        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1, 1), ylim=(-1, 1))
        ax.set_aspect('equal')
        # ax.grid()
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')

        # define geometries
        self._fig_geoms = []
        self._cart_patch = patches.Rectangle((0, 0), 0.3, 0.1, color=self._CART_COLOR)
        self._pole_patch, = ax.plot([], [], 'o-', lw=4, color=self._POLE_COLOR)
        self._fig_geoms.append(self._cart_patch)
        self._fig_geoms.append(self._pole_patch)

        # initialize plot
        ax.hlines(0,-1,1,'k',zorder=1) # zorder=1 makes it background
        ax.add_patch(self._cart_patch)
        ax.add_patch(self._pole_patch)
        
        # generate animation
        ani = animation.FuncAnimation(fig, self._animate_system, len(state_trjectory[0,:]), fargs=(state_trjectory,), interval=1, repeat=repeat)
        plt.show()
