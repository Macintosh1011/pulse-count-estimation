"""Generic pulse-count demonstration using matched filtering and MDL."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Simulation settings. Edit centers to change the count or introduce overlap.
N = 350
WIDTH = 15
CENTERS = [70, 175, 280]
AMPLITUDES = [1.0, 0.7, 0.5]
NOISE_STD = 0.0  # Start noiseless; try 0.03 next.
MAX_PULSES = 7


def estimate_count(received, width, max_pulses, *, return_diagnostics=False):
    """Fit candidate counts without knowing the true centers or amplitudes."""
    n = len(received)
    samples = np.arange(n)
    # Each column is the known pulse shape at a different candidate center.
    templates = np.exp(-0.5 * ((samples[:, None] - samples[None, :]) / width) ** 2)
    norms = np.linalg.norm(templates, axis=0)
    residual = received.copy()
    centers, scores, reconstructions, diagnostics = [], [], [], []
    # A shared numerical floor avoids log(0) for exact noiseless fits.
    floor = max(np.mean(received ** 2) * 1e-12, np.finfo(float).tiny)

    for k in range(1, max_pulses + 1):
        # Matched filtering: correlate the current residual with each template.
        matches = np.abs(templates.T @ residual) / norms
        matches[centers] = -np.inf
        centers.append(int(np.argmax(matches)))

        # Estimate amplitudes at the selected locations and subtract the fit.
        selected = templates[:, centers]
        amplitudes = np.linalg.lstsq(selected, received, rcond=None)[0]
        reconstruction = selected @ amplitudes
        residual = received - reconstruction
        rss = float(np.sum(residual ** 2))
        mse = max(rss / n, floor)

        # Matches the supplied MATLAB scripts: k delays, k amplitudes, variance.
        parameters = 2 * k + 1
        penalty = parameters * np.log(n)
        error_term = (n / 2) * np.log(mse)
        mdl = penalty + error_term
        diagnostics.append(dict(k=k, rss=rss, mse=rss / n, floor=floor,
                                floor_active=bool(rss / n <= floor),
                                penalty=float(penalty), error_term=float(error_term),
                                mdl=float(mdl), centers=centers.copy(),
                                amplitudes=amplitudes.tolist()))
        scores.append(mdl)
        reconstructions.append(reconstruction)

    best = int(np.argmin(scores))
    result = (best + 1, np.array(scores), reconstructions[best])
    return (*result, diagnostics) if return_diagnostics else result


def main():
    samples = np.arange(N)
    clean = sum(
        amplitude * np.exp(-0.5 * ((samples - center) / WIDTH) ** 2)
        for center, amplitude in zip(CENTERS, AMPLITUDES)
    )
    noise = np.random.default_rng(42).normal(0, NOISE_STD, N)
    received = clean + noise
    count, mdl, reconstruction = estimate_count(received, WIDTH, MAX_PULSES)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), layout="constrained")
    axes[0].plot(samples, received, color="#64748b", label="Received signal", linewidth=2)
    axes[0].plot(samples, reconstruction, "--", color="#7c3aed", label="Estimated reconstruction")
    axes[0].set(title=f"Generic Gaussian pulses: true count {len(CENTERS)}, estimated count {count}",
                xlabel="Sample index", ylabel="Amplitude")
    axes[0].legend()
    counts = np.arange(1, MAX_PULSES + 1)
    axes[1].plot(counts, mdl, "o-", color="#2563eb")
    axes[1].scatter(count, mdl[count - 1], color="#dc2626", s=65, zorder=3,
                    label=f"Minimum: {count} pulses")
    axes[1].set(xlabel="Assumed number of pulses, k", ylabel="Description length, DL(k)",
                xticks=counts)
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    output = Path(__file__).with_suffix(".png")
    fig.savefig(output, dpi=180)
    print(f"True count: {len(CENTERS)}; estimated count: {count}")
    print("Description lengths:", np.round(mdl, 3))
    print(f"Plot saved to {output}")
    if plt.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
