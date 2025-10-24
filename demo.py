import numpy as np
import matplotlib.pyplot as plt

def james_stein_positive(vec, sigma2):
    m = vec.size
    norm2 = np.sum(vec**2)
    if norm2 == 0:
        c_raw = -np.inf
    else:
        c_raw = 1.0 - (m - 2) * sigma2 / norm2
    c_pos = max(c_raw, 0.0)
    return c_pos * vec, c_pos, c_raw

def loss(x, theta):
    return 0.5 * np.sum((x - theta)**2)

def true_grad(x, theta):
    return x - theta

def run_trajectory(m, theta, x_init, sigma, lr_noisy, lr_js, lr_true, steps, rng):
    """
    Run three independent experiments for `steps` updates:
      - Noisy experiment: update with lr_noisy * (g_true_noisy + noise)
      - JS experiment: update with lr_js * JS_estimator(g_true_js + noise)
      - True experiment: update with lr_true * g_true_true (no noise)
    Each experiment computes gradient at its own current position.
    Noisy and JS experiments share the same noise sequence for fair comparison.
    Returns arrays: losses_noisy (len steps+1), losses_js, losses_true, c_pos_list, etc.
    """
    x_noisy = x_init.copy()
    x_js = x_init.copy()
    x_true = x_init.copy()
    sigma2 = sigma**2

    losses_noisy = [loss(x_noisy, theta)]
    losses_js = [loss(x_js, theta)]
    losses_true = [loss(x_true, theta)]
    c_pos_list = []
    c_raw_list = []
    norm_true_list = []
    norm_noisy_list = []
    norm_js_list = []
    norm_noise_list = []

    for t in range(steps):
        # 1. Generate shared noise (used for Noisy and JS experiments)
        noise = rng.normal(0.0, sigma, size=m)

        # 2. Noisy experiment (independent)
        g_true_noisy = true_grad(x_noisy, theta)
        G_noisy = g_true_noisy + noise
        x_noisy = x_noisy - lr_noisy * G_noisy
        losses_noisy.append(loss(x_noisy, theta))

        # 3. JS experiment (independent)
        g_true_js = true_grad(x_js, theta)
        G_js = g_true_js + noise  # Use the same noise
        JS_est, c_pos, c_raw = james_stein_positive(G_js, sigma2)
        x_js = x_js - lr_js * JS_est
        losses_js.append(loss(x_js, theta))

        # 4. True experiment (independent, no noise)
        g_true_true = true_grad(x_true, theta)
        x_true = x_true - lr_true * g_true_true
        losses_true.append(loss(x_true, theta))

        # 5. Record statistics
        c_pos_list.append(c_pos)
        c_raw_list.append(c_raw)
        norm_true_list.append(np.linalg.norm(g_true_true) / m)
        norm_noisy_list.append(np.linalg.norm(G_noisy) / m)
        norm_js_list.append(np.linalg.norm(JS_est) / m)
        norm_noise_list.append(np.linalg.norm(noise) / m)

    return np.array(losses_noisy), np.array(losses_js), np.array(losses_true), np.array(c_pos_list), np.array(c_raw_list), np.array(norm_true_list), np.array(norm_noisy_list), np.array(norm_js_list), np.array(norm_noise_list)

def tune_lrs(
    m=12,
    sigma=3.0,
    steps=10,
    seeds=(1,2,3,4,5),
    lr_noisy_grid=None,
    lr_js_grid=None
):
    """
    Independent grid search over lr_noisy and lr_js.
    For each pair, run with multiple seeds and compute avg final loss of JS trajectory.
    Choose the pair that minimizes avg final JS final loss.
    """
    if lr_noisy_grid is None:
        lr_noisy_grid = np.linspace(0.02, 0.28, 100)  # candidate lr for noisy trajectory
    if lr_js_grid is None:
        lr_js_grid = np.linspace(0.05, 0.6, 100)    # candidate lr for JS trajectory (independent)

    best = None
    results = []

    for lr_noisy in lr_noisy_grid:
        for lr_js in lr_js_grid:
            final_losses = []
            for sd in seeds:
                rng = np.random.RandomState(sd)
                # non-constant theta and x_init
                theta = 0.5 * np.sin(np.linspace(0, 3*np.pi, m))
                x_init = 1.5 + 0.8 * np.cos(np.linspace(0, 2*np.pi, m))
                losses_noisy, losses_js, _, _, _, _, _, _, _ = run_trajectory(m, theta, x_init, sigma, lr_noisy, lr_js, 0.1, steps, rng)
                final_losses.append(losses_js[-1])
            avg_final_js_loss = float(np.mean(final_losses))
            results.append((lr_noisy, lr_js, avg_final_js_loss))
            if (best is None) or (avg_final_js_loss < best[2]):
                best = (lr_noisy, lr_js, avg_final_js_loss)
    return best, results

def tune_lr_true(
    m=12,
    sigma=3.0,
    steps=10,
    seeds=(1,2,3,4,5),
    lr_true_grid=None
):
    """
    Grid search for optimal learning rate for true gradient trajectory.
    """
    if lr_true_grid is None:
        lr_true_grid = np.linspace(0.01, 0.2, 100)  # candidate lr for true gradient trajectory
    
    best = None
    results = []
    
    for lr_true in lr_true_grid:
        final_losses = []
        for sd in seeds:
            rng = np.random.RandomState(sd)
            theta = 0.5 * np.sin(np.linspace(0, 3*np.pi, m))
            x_init = 1.5 + 0.8 * np.cos(np.linspace(0, 2*np.pi, m))
            # Use dummy values for lr_noisy and lr_js since we only care about true trajectory
            losses_noisy, losses_js, losses_true, _, _, _, _, _, _ = run_trajectory(m, theta, x_init, sigma, 0.1, 0.2, lr_true, steps, rng)
            final_losses.append(losses_true[-1])
        avg_final_true_loss = float(np.mean(final_losses))
        results.append((lr_true, avg_final_true_loss))
        if (best is None) or (avg_final_true_loss < best[1]):
            best = (lr_true, avg_final_true_loss)
    return best, results

if __name__ == "__main__":
    # -------------------- user-tunable parameters --------------------
    m = 10000
    sigma = 3.0
    steps = 10
    seeds = (999,)   # single seed for tuning
    # lr grids (you can adjust ranges)
    lr_noisy_grid = np.linspace(0.001, 0.01, 100)
    lr_js_grid = np.linspace(0.05, 0.6, 100)  # independent grid for lr_js
    # ----------------------------------------------------------------

    print("Tuning learning rates (this may take a few seconds)...")
    best, all_results = tune_lrs(
        m=m,
        sigma=sigma,
        steps=steps,
        seeds=seeds,
        lr_noisy_grid=lr_noisy_grid,
        lr_js_grid=lr_js_grid
    )

    if best is None:
        print("Tuning failed.")
        raise SystemExit(1)

    best_lr_noisy, best_lr_js, best_avg_js_loss = best
    print(f"Best pair (by avg final JS loss over seeds): lr_noisy = {best_lr_noisy:.4f}, lr_js = {best_lr_js:.4f}, avg_final_JS_loss = {best_avg_js_loss:.6f}")

    print("Tuning learning rate for true gradient trajectory...")
    best_true, all_results_true = tune_lr_true(
        m=m,
        sigma=sigma,
        steps=steps,
        seeds=seeds,
        lr_true_grid=np.linspace(0.01, 0.2, 100)
    )

    if best_true is None:
        print("Tuning for true gradient failed.")
        raise SystemExit(1)

    best_lr_true, best_avg_true_loss = best_true
    print(f"Best lr for true gradient: lr_true = {best_lr_true:.4f}, avg_final_loss = {best_avg_true_loss:.6f}")

    # Run a final illustrative run with the best pair using a fresh seed and display plots
    seed_final = 999
    rng = np.random.RandomState(seed_final)
    theta = 0.5 * np.sin(np.linspace(0, 3*np.pi, m))
    x_init = 1.5 + 0.8 * np.cos(np.linspace(0, 2*np.pi, m))

    losses_noisy, losses_js, losses_true, c_pos_list, c_raw_list, norm_true_list, norm_noisy_list, norm_js_list, norm_noise_list = run_trajectory(
        m, theta, x_init, sigma, best_lr_noisy, best_lr_js, best_lr_true, steps, rng
    )

    # Plot 1: Loss per step
    steps_range = np.arange(0, steps+1)
    plt.figure(figsize=(15,4))
    plt.subplot(1,3,1)
    plt.plot(steps_range, losses_noisy, marker='o', label=f'noisy-grad (lr={best_lr_noisy:.3f})')
    plt.plot(steps_range, losses_js, marker='o', label=f'JS-grad (lr={best_lr_js:.3f})')
    plt.plot(steps_range, losses_true, marker='s', label=f'true-grad (lr={best_lr_true:.3f})')
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Loss per step (final run with best lr pair)")
    plt.legend()
    plt.grid(True)

    # Plot 2: JS shrinkage coefficient per step
    plt.subplot(1,3,2)
    plt.plot(np.arange(1, steps+1), c_pos_list, marker='s', label='c_pos (JS positive-part)')
    plt.plot(np.arange(1, steps+1), c_raw_list, marker='x', label='c_raw (raw)')
    plt.xlabel("step")
    plt.ylabel("shrinkage coefficient")
    plt.title("JS coefficients per step (final run)")
    plt.ylim(-0.5, 1.05)
    plt.legend()
    plt.grid(True)

    # Plot 3: Gradient L2 norm per step
    plt.subplot(1,3,3)
    steps_range_bar = np.arange(1, steps+1)
    width = 0.2
    plt.bar(steps_range_bar - 1.5*width, norm_true_list, width, label='true grad norm/m', alpha=0.8)
    plt.bar(steps_range_bar - 0.5*width, norm_noisy_list, width, label='noisy grad norm/m', alpha=0.8)
    plt.bar(steps_range_bar + 0.5*width, norm_js_list, width, label='JS grad norm/m', alpha=0.8)
    plt.bar(steps_range_bar + 1.5*width, norm_noise_list, width, label='noise norm/m', alpha=0.8)
    plt.xlabel("step")
    plt.ylabel("Average norm per dimension")
    plt.title("Gradient norm per dimension (per step)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Print short summary
    print("\nFinal run summary (per-step):")
    for t in range(steps):
        print(f" step {t+1:2d}: c_pos={c_pos_list[t]:.6f}, c_raw={c_raw_list[t]:.6f}, loss_noisy={losses_noisy[t+1]:.6f}, loss_js={losses_js[t+1]:.6f}, loss_true={losses_true[t+1]:.6f}, norm_true/m={norm_true_list[t]:.6f}, norm_noisy/m={norm_noisy_list[t]:.6f}, norm_js/m={norm_js_list[t]:.6f}, norm_noise/m={norm_noise_list[t]:.6f}")

    print("\nDone.")