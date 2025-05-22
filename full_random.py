import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit  # sigmoid
from collections import defaultdict

# ---------- Parameters ----------
np.random.seed(45)
num_items = 10
num_users = 5
num_rounds = 1000
num_step = 5
start_update_theta_after = 20
learning_rate = 1
momentum = 0.9
num_observations_per_round = 50

# Ground truth
true_s = np.random.randn(num_items)
true_g = np.random.randint(0, 2, size=num_items)
true_theta = np.random.randn(num_users)

def true_p_ij(s_i, s_j):
    return np.exp(s_i) / (np.exp(s_i) + np.exp(s_j))

def biased_p_ij(i, j, k):
    si, sj = true_s[i], true_s[j]
    gi, gj = true_g[i], true_g[j]
    ti = true_theta[k]
    return np.exp(si + ti * gi) / (np.exp(si + ti * gi) + np.exp(sj + ti * gj))

def compute_unbiased_probs(s):
    pij = np.zeros((num_items, num_items))
    for i in range(num_items):
        for j in range(num_items):
            if i != j:
                pij[i, j] = true_p_ij(s[i], s[j])
    return pij

def compute_true_unbiased_probs():
    pij = np.zeros((num_items, num_items))
    for i in range(num_items):
        for j in range(num_items):
            if i != j:
                pij[i, j] = true_p_ij(true_s[i], true_s[j])
    return pij

def update_group_assignments(s, theta, p_tot, p_cnt, g):
    new_g = g.copy()
    for i in range(num_items):
        losses = []
        for trial_gi in [0, 1]:
            g_trial = g.copy()
            g_trial[i] = trial_gi
            loss = 0.0
            for k in range(num_users):
                for j in range(num_items):
                    if i == j:
                        continue
                    gi, gj = g_trial[i], g_trial[j]
                    gamma_i = np.exp(theta[k] * gi)
                    gamma_j = np.exp(theta[k] * gj)
                    s_i, s_j = s[i], s[j]
                    pk = gamma_i * np.exp(s_i) / (gamma_i * np.exp(s_i) + gamma_j * np.exp(s_j))
                    observed = p_cnt[(i, j, k)] / (p_tot[(i, j, k)] + 1e-8)
                    loss += (pk - observed) ** 2
            losses.append(loss)
        new_g[i] = 0 if losses[0] < losses[1] else 1
    return new_g

def update_parameters(s, theta, g, counts, totals, learning_rate=0.01, update_theta=True, momentum=0.9):
    momentum_s = np.zeros_like(s)
    if update_theta:
        momentum_theta = np.zeros_like(theta)
    for _ in range(num_step):
        grad_s = np.zeros(num_items)
        if update_theta:
            grad_theta = np.zeros(num_users)
        for k in range(num_users):
            for i in range(num_items):
                for j in range(num_items):
                    if i == j:
                        continue
                    n_ij = totals[(i, j, k)]
                    if n_ij == 0:
                        continue
                    freq = counts[(i, j, k)] / n_ij
                    si, sj = s[i], s[j]
                    gi, gj = g[i], g[j]
                    th = theta[k]
                    exp1 = np.exp(si + th * gi)
                    exp2 = np.exp(sj + th * gj)
                    pred = exp1 / (exp1 + exp2)
                    err = pred - freq
                    grad_s[i] += err * pred * (1 - pred)
                    grad_s[j] -= err * pred * (1 - pred)
                    if update_theta:
                        grad_theta[k] += err * pred * (1 - pred) * (gi - gj)
        momentum_s = momentum * momentum_s + (1 - momentum) * grad_s
        s -= learning_rate * momentum_s
        if update_theta:
            momentum_theta = momentum_theta * momentum + (1 - momentum) * grad_theta
            theta -= learning_rate * momentum_theta
    return s, theta

# ---------- Initialization ----------
s = np.random.randn(num_items)
theta = np.random.randn(num_users)
g = np.random.randint(0, 2, size=num_items)

counts = defaultdict(int)
totals = defaultdict(int)

errors_p = []
errors_theta = []

# ---------- Round-by-round Simulation ----------
for round in range(num_rounds):
    for k in range(num_users):
        pairs = []
        while len(pairs) < num_observations_per_round:
            i, j = np.random.choice(num_items, 2, replace=False)
            if i != j and (i, j) not in pairs:
                pairs.append((i, j))
        for i, j in pairs:
            pij = biased_p_ij(i, j, k)
            if np.random.rand() < pij:
                counts[(i, j, k)] += 1
            else:
                counts[(j, i, k)] += 1
            totals[(i, j, k)] += 1
            totals[(j, i, k)] += 1

    if round < start_update_theta_after:
        s, theta = update_parameters(s, theta, g, counts, totals, learning_rate=learning_rate, update_theta=False, momentum=momentum)
    else:
        s, theta = update_parameters(s, theta, g, counts, totals, learning_rate=learning_rate, update_theta=True, momentum=momentum)
    g = update_group_assignments(s, theta, totals, counts, g)

    est_pij = compute_unbiased_probs(s)
    true_pij = compute_true_unbiased_probs()
    error_p = np.mean(np.abs(est_pij - true_pij))
    error_theta = np.mean(np.abs(theta - true_theta))
    errors_p.append(error_p)
    errors_theta.append(error_theta)

# ---------- Results ----------
print("Final estimated s vs true s:")
print(f"Estimated s: \n{s}\n, True s: \n{true_s}\n")
print(f"Estimated theta: \n{theta}\n{true_g}\n, True theta: \n{true_theta}\n{true_g}\n")
print(f"error in estimated p_ij: {errors_p[-1]:.4f}")
print(f"error in estimated theta: {errors_theta[-1]:.4f}")

# ---------- Plot ----------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(errors_p, label="Unbiased p_ij MSE")
plt.xlabel("Round")
plt.ylabel("MSE")
plt.title("Error in estimated p_ij")
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(errors_theta, label="Theta MSE", color='orange')
plt.xlabel("Round")
plt.ylabel("MSE")
plt.title("Error in estimated theta")
plt.grid()
plt.legend()

plt.tight_layout()
plt.savefig('estimation_errors_random_obs.png')