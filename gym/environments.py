# typehinting
from typing import Callable
# arrays
import numpy as np
# visualization
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class CartPoleSystem:
    def __init__(self, initial_condition:np.ndarray):

        # === system dynamic realted properties ===
        self._parameter = np.array([
            0.8,      # lp: length of the pole
            0.1,      # mp: mass of the pole
            0.01,      # Jp: moment of interia of the pole
            0.5,      # mc: mass of the cart
            9.81,   # g: acceleration due to gravity
        ])
        self._nx = 4 # state dim
        self._nu = 1 # input dim
        self._samplerate = 0.01
        self._state = initial_condition

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

        self._CART_COLOR = 'k'


    @property
    def samplerate(self):
        return self._samplerate

    @samplerate.setter
    def samplerate(self, samplerate : float):
        self._samplerate = samplerate
        
    def _f(self, x:np.ndarray, u:float) -> np.ndarray:
        '''
        Evaluates the vectorfield.
        '''
        # read state
        s, phi, ds, dphi = x
        # read parameter
        lp, mp, Jp, mc, g = self._parameter
        a = lp/2
        # evaluate vectorfield
        tmp = a*mp/(Jp+2*a*mp)
        return np.array([
            ds,
            dphi,
            u,
            tmp*g*np.sin(phi) + tmp*np.cos(phi)*u
        ])

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

    def draw(self, figure_number : int):
        plt.figure(figure_number)
        plt.clf()

        # draw line
        plt.hlines(0,-1,1,'k')
        
        s, phi, ds, dphi = self._state
        lp, _, _, _, _ = self._parameter

        # draw cart
        plt.gca().add_patch(Rectangle((s-self._CART_WIDTH/2,0-self._CART_HIGHT/2),self._CART_WIDTH,self._CART_HIGHT,color=self._CART_COLOR))

        # draw pole
        x_pole = (s, s - np.sin(phi)*lp)
        y_pole = (0, np.cos(phi)*lp)
        plt.plot(x_pole,y_pole,color = 'C1', linewidth=3)

        plt.gca().set_xlim([-1,1])
        plt.gca().set_ylim([-1,1])

        # plt.axis('equal')
        plt.pause(0.01)

