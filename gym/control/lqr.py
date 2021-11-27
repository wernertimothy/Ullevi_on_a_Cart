# typehinting
from typing import Tuple
# arrays
import numpy as np
# Ricatti
import scipy.linalg

def lqr(
        A : np.ndarray,      # state matrix
        B : np.ndarray,      # input matrix
        Q : np.ndarray,      # state penalty
        R : np.ndarray,      # input penlaty
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    calculates the static feedback gain K s.t. u = -Kx.
    '''
    # solve continuous Riccati
    P = np.array(scipy.linalg.solve_continuous_are(A, B, Q, R))
    # calculate LQR gain
    # K = np.array(scipy.linalg.inv(R)*(B.T*P))
    K = np.array(B.T@P)/R
    # calculate closed-loop eigenvalues
    eig = np.zeros(4)

    return K, P, eig

          