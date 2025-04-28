import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(42)

def generate_true_params(M):
    theta_i = np.random.randn()
    theta_j = np.random.randn()
    p_ij_true = np.exp(theta_i) / (np.exp(theta_i) + np.exp(theta_j))
    r_true = np.random.uniform(0.5, 1.5, size=M)
    return theta_i, theta_j, p_ij_true, r_true

def simulate_observations(theta_i, theta_j, r_true, M):
    wins = np.zeros(M)
    trials = np.zeros(M)
    for m in range(M):
        p_m = np.exp(theta_i) / (np.exp(theta_i) + r_true[m] * np.exp(theta_j))
        outcome = np.random.rand() < p_m
        wins[m] += outcome
        trials[m] += 1
    return wins, trials

def estimate_parameters(wins, trials, r_hat, p_ij_hat, k, t, lr=0.1, steps=10):
    M = len(wins)
    p_minus_m = wins / trials

    if t < k:
        p_ij_hat = np.mean(p_minus_m)
    else:
        for _ in range(steps):
            r_hat = update_r_hat(p_minus_m, p_ij_hat, r_hat)
            p_ij_hat = update_p_hat(p_minus_m, r_hat, lr)
    return p_ij_hat, r_hat

def update_r_hat(p_minus_m, p_ij_hat, r_hat):
    M = len(p_minus_m)
    new_r_hat = np.copy(r_hat)
    pm = np.clip(p_minus_m, 1e-3, 1-1e-3)
    for m in range(M):
        new_r_hat[m] = p_ij_hat * (1 - pm[m]) / ( pm[m] * (1 -  p_ij_hat))
        new_r_hat[m] = np.clip(new_r_hat[m], 0.1, 2.0)
    return new_r_hat

def update_p_hat(p_minus_m, r_hat, lr):
    p = 0.5
    for _ in range(50):
        grad = 0
        for m in range(len(r_hat)):
            denom = (1 - r_hat[m]) * p + r_hat[m]
            pred = p / denom
            grad += 2 * (pred - p_minus_m[m]) * (denom + (r_hat[m] - 1) * p) / (denom**2)
        p -= lr * grad
        p = np.clip(p, 1e-3, 1 - 1e-3)
    return p

def simulate_one_run(M=10, num_rounds=500, k=20):
    theta_i, theta_j, p_ij_true, r_true = generate_true_params(M)
    wins = np.zeros(M)
    trials = np.zeros(M)
    r_hat = np.ones(M)
    p_ij_hat = 0.5

    errors_p = []
    errors_r = []

    for t in range(1, num_rounds + 1):
        round_wins, round_trials = simulate_observations(theta_i, theta_j, r_true, M)
        wins += round_wins
        trials += round_trials

        p_ij_hat, r_hat = estimate_parameters(wins, trials, r_hat, p_ij_hat, k, t)

        error_p = abs(p_ij_hat - p_ij_true)
        error_r = np.mean(np.abs(r_hat - r_true))

        errors_p.append(error_p)
        errors_r.append(error_r)

        if t == num_rounds:
            print(f"p_ij: {p_ij_hat:.4f}, p_true: {p_ij_true:.4f}")

    return errors_p, errors_r

def plot_results(mean_errors_p, mean_errors_r):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(mean_errors_p)
    plt.xlabel('Rounds')
    plt.ylabel('Error in $p_{ij}$')
    plt.title('Mean Error in $p_{ij}$ over Rounds')

    plt.subplot(1, 2, 2)
    plt.plot(mean_errors_r)
    plt.xlabel('Rounds')
    plt.ylabel('Mean Error in $r_m$')
    plt.title('Mean Error in $r_m$ over Rounds')

    plt.tight_layout()
    plt.show()

def main():
    num_runs = 5
    M = 5
    num_rounds = 1000
    k = (M * M)/3

    all_errors_p = []
    all_errors_r = []

    for i in range(num_runs):
        time_st = time.time()
        errors_p, errors_r = simulate_one_run(M, num_rounds, k)
        all_errors_p.append(errors_p)
        all_errors_r.append(errors_r)
        time_ed = time.time()
        print(f"Run {i + 1} completed, time used: {time_ed - time_st:.4f} seconds")
        print(f"error in p_ij: {errors_p[-1]:.4f}, error in r_m: {errors_r[-1]:.4f}")

    all_errors_p = np.array(all_errors_p)
    all_errors_r = np.array(all_errors_r)

    mean_errors_p = np.mean(all_errors_p, axis=0)
    mean_errors_r = np.mean(all_errors_r, axis=0)

    plot_results(mean_errors_p, mean_errors_r)

    print(f"Final mean error in p_ij: {mean_errors_p[-1]:.4f}")
    print(f"Final mean error in r_m: {mean_errors_r[-1]:.4f}")

if __name__ == "__main__":
    main()
