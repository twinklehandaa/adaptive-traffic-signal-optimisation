"""dqn_agent.py : Deep Q-Network agent.

Implemented directly in NumPy (manual forward pass + backprop + Adam)
so the agent has zero heavyweight ML-framework dependency and trains
fast on CPU. Architecturally it is exactly the network described in the
report: a 2-hidden-layer fully-connected Q(s,a) approximator, an
experience replay buffer, a periodically-synced target network and
epsilon-greedy exploration -- only the tensor backend differs from the
PyTorch reference implementation in the report.
"""
import random
from collections import deque

import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(x.dtype)


class QNetwork:
    """3-layer fully connected network approximating Q_theta(s, a)."""

    def __init__(self, state_dim, action_dim, hidden=64, rng=None):
        rng = rng or np.random.default_rng(0)
        s = lambda n_in: np.sqrt(2.0 / n_in)
        self.W1 = rng.normal(0, s(state_dim), (state_dim, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.normal(0, s(hidden), (hidden, hidden)).astype(np.float32)
        self.b2 = np.zeros(hidden, dtype=np.float32)
        self.W3 = rng.normal(0, s(hidden), (hidden, action_dim)).astype(np.float32)
        self.b3 = np.zeros(action_dim, dtype=np.float32)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def set_params(self, params):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = [p.copy() for p in params]

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)
        q = a2 @ self.W3 + self.b3
        cache = (x, z1, a1, z2, a2)
        return q, cache

    def backward(self, cache, dq):
        x, z1, a1, z2, a2 = cache
        n = x.shape[0]
        dW3 = a2.T @ dq / n
        db3 = dq.mean(axis=0)
        da2 = dq @ self.W3.T
        dz2 = da2 * relu_grad(z2)
        dW2 = a1.T @ dz2 / n
        db2 = dz2.mean(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_grad(z1)
        dW1 = x.T @ dz1 / n
        db1 = dz1.mean(axis=0)
        return [dW1, db1, dW2, db2, dW3, db3]


class Adam:
    def __init__(self, params, lr=5e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads, clip=10.0):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            g = np.clip(g, -clip, clip)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=5e-4, gamma=0.95,
                 buffer=20_000, batch=64, target_sync=300, hidden=64, seed=0):
        rng = np.random.default_rng(seed)
        self.q = QNetwork(state_dim, action_dim, hidden, rng)
        self.target = QNetwork(state_dim, action_dim, hidden, rng)
        self.target.set_params(self.q.params())
        self.opt = Adam(self.q.params(), lr=lr)
        self.memory = deque(maxlen=buffer)
        self.gamma, self.batch, self.target_sync = gamma, batch, target_sync
        self.action_dim, self.learn_step = action_dim, 0
        self.eps, self.eps_min, self.eps_decay = 1.0, 0.05, 0.975
        self.rng = rng

    def act(self, state):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.action_dim))
        q, _ = self.q.forward(state[None, :])
        return int(np.argmax(q[0]))

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, float(done)))

    def replay(self):
        if len(self.memory) < self.batch:
            return None
        batch = random.sample(self.memory, self.batch)
        s, a, r, s2, dn = map(np.array, zip(*batch))

        q_pred, cache = self.q.forward(s)
        q_next, _ = self.target.forward(s2)
        q_target = q_pred.copy()
        max_next = q_next.max(axis=1)
        for i in range(len(a)):
            q_target[i, a[i]] = r[i] + self.gamma * max_next[i] * (1 - dn[i])

        diff = q_pred - q_target  # dLoss/dq_pred for smooth-l1-ish (use MSE grad)
        loss = float(np.mean(diff ** 2))
        grads = self.q.backward(cache, 2 * diff)
        self.opt.step(self.q.params(), grads)

        self.learn_step += 1
        if self.learn_step % self.target_sync == 0:
            self.target.set_params(self.q.params())
        return loss

    def decay_epsilon(self):
        """Called once per episode (not per step) so exploration is spread
        across the whole training run instead of collapsing in episode 1."""
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def save(self, path):
        np.savez(path, W1=self.q.W1, b1=self.q.b1, W2=self.q.W2, b2=self.q.b2,
                  W3=self.q.W3, b3=self.q.b3, eps=self.eps)

    def load(self, path):
        d = np.load(path)
        self.q.set_params([d['W1'], d['b1'], d['W2'], d['b2'], d['W3'], d['b3']])
        self.eps = float(d['eps'])
