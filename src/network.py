"""One-hidden-layer network, sigmoid in both layers, bias units fixed at -1.

Trained by SGD with momentum and weight decay. Also computes the output
sensitivity S_ki and the informativeness that SASLA needs.
"""

from __future__ import annotations

import numpy as np


def sigmoid(net: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(net, -60.0, 60.0)))


class FFNN:
    """One hidden layer feedforward network.

    Weights are drawn uniformly from [-1/sqrt(fanin), 1/sqrt(fanin)], which
    keeps initial net input signals near zero where the derivative of the
    sigmoid function is largest, avoiding premature convergence.
    """

    def __init__(self, n_in: int, n_hidden: int, n_out: int, rng: np.random.Generator):
        self.I, self.J, self.K = n_in, n_hidden, n_out
        b_v = 1.0 / np.sqrt(n_in + 1)
        b_w = 1.0 / np.sqrt(n_hidden + 1)
        self.V = rng.uniform(-b_v, b_v, (n_hidden, n_in + 1))
        self.W = rng.uniform(-b_w, b_w, (n_out, n_hidden + 1))
        self.dV = np.zeros_like(self.V)
        self.dW = np.zeros_like(self.W)

    # --- forward

    def hidden(self, Z: np.ndarray) -> np.ndarray:
        """Hidden activations with bias column appended. Z is (n, I+1)."""
        H = sigmoid(Z @ self.V.T)
        out = np.empty((H.shape[0], self.J + 1))
        out[:, :-1] = H
        out[:, -1] = -1.0
        return out

    def forward(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Hb = self.hidden(Z)
        return Hb, sigmoid(Hb @ self.W.T)

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return self.forward(Z)[1]

    # --- training

    def train_epoch(self, Z: np.ndarray, T: np.ndarray, eta: float, alpha: float,
                    lam: float, rng: np.random.Generator) -> int:
        """One epoch of stochastic learning over the given training subset.

        Weights are updated after every pattern presentation, and the subset
        is shuffled first so that the network cannot learn the order in which
        patterns are presented. Returns the number of pattern presentations.
        """
        V, W = self.V, self.W
        dV, dW = self.dV, self.dW
        J = self.J
        yb = np.empty(J + 1)
        yb[-1] = -1.0
        Wh = W[:, :J]

        for p in rng.permutation(Z.shape[0]):
            z = Z[p]
            yb[:J] = sigmoid(V @ z)
            o = sigmoid(W @ yb)

            d_o = (T[p] - o) * o * (1.0 - o)                    # output error signals
            d_y = yb[:J] * (1.0 - yb[:J]) * (Wh.T @ d_o)        # hidden error signals

            # gradient of E_T + lam * E_C, with E_C = 0.5 * sum of squared weights
            np.multiply(dW, alpha, out=dW)
            dW += eta * (np.outer(d_o, yb) - lam * W)
            W += dW

            np.multiply(dV, alpha, out=dV)
            dV += eta * (np.outer(d_y, z) - lam * V)
            V += dV

        return Z.shape[0]

    # --- sensitivity

    def sensitivity(self, Z: np.ndarray) -> np.ndarray:
        """Output-input layer sensitivity matrix of Engelbrecht (2001).

        For pattern p, element (k, i) is the derivative of output o_k with
        respect to input z_i,

            S_ki = o_k (1 - o_k) * sum_j w_kj y_j (1 - y_j) v_ji ,

        for sigmoid activation functions in both layers. Bias units are
        excluded, because a bias unit holds a constant value that is never
        perturbed. Returns an array of shape (n, K, I).
        """
        Hb, O = self.forward(Z)
        H = Hb[:, :self.J]
        dO = O * (1.0 - O)                      # (n, K)
        dH = H * (1.0 - H)                      # (n, J)
        Wh = self.W[:, :self.J]                 # (K, J)
        Vh = self.V[:, :self.I]                 # (J, I)
        # inner[n, k, i] = sum_j Wh[k, j] * dH[n, j] * Vh[j, i]
        inner = np.einsum("kj,nj,ji->nki", Wh, dH, Vh, optimize=True)
        return dO[:, :, None] * inner

    def informativeness(self, Z: np.ndarray) -> np.ndarray:
        """Pattern informativeness of Engelbrecht (2001).

        The sensitivity matrix of every pattern is collapsed across inputs
        with the Euclidean norm and across outputs with the maximum norm.
        """
        S = self.sensitivity(Z)
        S_ok = np.sqrt(np.sum(S * S, axis=2))   # Euclidean norm over inputs
        return np.max(np.abs(S_ok), axis=1)     # maximum norm over outputs

    # --- state

    def snapshot(self):
        return self.V.copy(), self.W.copy()

    def restore(self, state):
        self.V[...], self.W[...] = state
        self.dV.fill(0.0)
        self.dW.fill(0.0)

    @property
    def n_weights(self) -> int:
        return self.V.size + self.W.size
