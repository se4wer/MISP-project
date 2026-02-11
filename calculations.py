"""
Generalized Tikhonov Regularization & Error Analysis
Mathematics for Imaging and Signal Processing — A.A. 2025/2026

This script contains all computational routines and runs the full analysis:
  Part 1 — Forward problem (blur + noise simulation)
  Part 2 — Generalized Tikhonov regularization (L², H¹, H² penalties)
  Part 3 — Spectral windowing reconstruction
  Part 4 — Bias-variance trade-off analysis
"""

import numpy as np
from numpy.fft import fft2, ifft2, ifftshift
from scipy.special import j1

np.random.seed(42)


# ============================================================
# Utility
# ============================================================

def compute_psnr(original, reconstructed):
    """Peak signal-to-noise ratio in dB."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse < 1e-15:
        return float('inf')
    peak_val = np.max(original)
    return 10 * np.log10(peak_val ** 2 / mse)


# ============================================================
# Test image
# ============================================================

def shepp_logan_phantom(N=256):
    """Generate the Shepp-Logan phantom (N x N).
    Sharp elliptical boundaries at various angles make it ideal for
    evaluating edge preservation under different regularization penalties.
    """
    img = np.zeros((N, N), dtype=np.float64)
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)

    # (intensity, semi-axis a, semi-axis b, centre x0, y0, rotation deg)
    ellipses = [
        ( 1.0,   0.6900, 0.9200,  0.0,     0.0,     0),
        (-0.8,   0.6624, 0.8740,  0.0,    -0.0184,  0),
        (-0.2,   0.1100, 0.3100, -0.22,    0.0,    -18),
        (-0.2,   0.1600, 0.4100,  0.22,    0.0,     18),
        ( 0.1,   0.2100, 0.2500,  0.0,     0.35,    0),
        ( 0.1,   0.0460, 0.0460,  0.0,     0.1,     0),
        ( 0.1,   0.0460, 0.0460,  0.0,    -0.1,     0),
        ( 0.1,   0.0460, 0.0230, -0.08,   -0.605,   0),
        ( 0.1,   0.0230, 0.0230,  0.0,    -0.606,   0),
        ( 0.1,   0.0230, 0.0460,  0.06,   -0.605,   0),
    ]

    for rho, a, b, x0, y0, theta_deg in ellipses:
        theta = np.radians(theta_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        Xr =  cos_t * (X - x0) + sin_t * (Y - y0)
        Yr = -sin_t * (X - x0) + cos_t * (Y - y0)
        mask = (Xr / a) ** 2 + (Yr / b) ** 2 <= 1
        img[mask] += rho

    return img


# ============================================================
# Part 1 — Forward Problem
# ============================================================

def forward_problem(f_true, K_hat, snr_db):
    """Algorithm 2: simulate blur + noise.
    Blur in Fourier domain (pointwise multiply), then add Gaussian
    noise calibrated to the desired SNR (dB).
    """
    f_hat = fft2(f_true)
    g_hat_clean = f_hat * K_hat
    g_clean = np.real(ifft2(g_hat_clean))

    sigma_noise = 10 ** (-snr_db / 20) * np.std(g_clean)
    noise = np.random.normal(0, sigma_noise, f_true.shape)

    g = g_clean + noise
    g_hat = fft2(g)
    return g, g_hat, g_clean, sigma_noise, noise


# ============================================================
# Part 2 — Generalized Tikhonov Regularization
# ============================================================

def tikhonov_reconstruct(g_hat, K_hat, P_sq, mu):
    """Algorithm 3: Generalized Tikhonov reconstruction.
    f_hat_mu = K_hat* / (|K_hat|^2 + mu * |P|^2) * g_hat
    """
    denominator = np.abs(K_hat) ** 2 + mu * P_sq
    f_hat_mu = np.conj(K_hat) * g_hat / denominator
    return np.real(ifft2(f_hat_mu))


# ============================================================
# Part 3 — Spectral Windowing
# ============================================================

def spectral_window_reconstruct(g_hat, K_hat, Omega_cutoff, omega_mag_fft):
    """Hard frequency cutoff reconstruction.
    Keep frequencies where |omega| < Omega, set everything else to zero.
    """
    W = (omega_mag_fft < Omega_cutoff).astype(float)
    f_hat_Omega = np.zeros_like(g_hat)
    mask = W > 0
    f_hat_Omega[mask] = W[mask] * g_hat[mask] / K_hat[mask]
    return np.real(ifft2(f_hat_Omega))


# ============================================================
# Part 4 — Bias-Variance Trade-off
# ============================================================

def compute_bias_variance(K_hat, P_sq, f_true, sigma_noise, mu_values):
    """Compute bias^2, variance, and total error for each mu.
    Computed in the Fourier domain via Parseval's identity.
    """
    N_pix = K_hat.shape[0] * K_hat.shape[1]
    f_hat_true = fft2(f_true)
    f_hat_sq = np.abs(f_hat_true) ** 2
    K_hat_sq = np.abs(K_hat) ** 2

    bias_vals = np.zeros(len(mu_values))
    var_vals = np.zeros(len(mu_values))

    for i, mu in enumerate(mu_values):
        denom = K_hat_sq + mu * P_sq
        R_mu = np.conj(K_hat) / denom

        # Bias: how far is R_mu * K from the identity
        R_K_minus_1 = R_mu * K_hat - 1.0
        bias_vals[i] = (1.0 / N_pix) * np.sum(np.abs(R_K_minus_1) ** 2 * f_hat_sq)

        # Variance: how much noise gets through the filter
        var_vals[i] = sigma_noise ** 2 * np.sum(np.abs(R_mu) ** 2)

    total_vals = bias_vals + var_vals
    return bias_vals, var_vals, total_vals


# ============================================================
# Run full analysis
# ============================================================

if __name__ == '__main__':

    N = 256
    f_true = shepp_logan_phantom(N)

    # --- Algorithm 1: frequency grid ---
    M = N
    x_range = np.arange(-N // 2, N // 2)
    y_range = np.arange(-M // 2, M // 2)
    omega_X, omega_Y = np.meshgrid(x_range, y_range)
    omega_mag = np.sqrt(omega_X ** 2 + omega_Y ** 2)

    # --- Blur kernels (centred then ifftshift to FFT layout) ---

    # Gaussian
    sigma_blur = 3.0
    K_hat_gauss_c = np.exp(-omega_mag ** 2 / (2 * sigma_blur ** 2))
    K_hat_gauss = ifftshift(K_hat_gauss_c)

    # Linear motion blur (horizontal, L pixels)
    L_motion = 15
    motion_arg = omega_X * L_motion / 2
    K_hat_motion_c = (
        np.exp(-1j * (L_motion / 2) * omega_X)
        * np.sinc(motion_arg / np.pi)
    )
    K_hat_motion = ifftshift(K_hat_motion_c)

    # Out-of-focus (circular aperture, radius R)
    R_defocus = 8.0
    defocus_arg = omega_mag * R_defocus
    with np.errstate(divide='ignore', invalid='ignore'):
        K_hat_oof_c = np.where(
            defocus_arg > 1e-10,
            2 * j1(defocus_arg) / defocus_arg,
            1.0
        )
    K_hat_oof = ifftshift(K_hat_oof_c)

    # --- Forward problem: generate observations ---

    g_high, g_hat_high, g_clean, sigma_high, noise_high = forward_problem(f_true, K_hat_gauss, 40)
    g_low, g_hat_low, _, sigma_low, noise_low = forward_problem(f_true, K_hat_gauss, 20)

    g_motion_high, g_hat_motion_high, _, sigma_motion_high, _ = forward_problem(f_true, K_hat_motion, 40)
    g_motion_low, g_hat_motion_low, _, sigma_motion_low, _ = forward_problem(f_true, K_hat_motion, 20)

    g_oof_high, g_hat_oof_high, _, sigma_oof_high, _ = forward_problem(f_true, K_hat_oof, 40)
    g_oof_low, g_hat_oof_low, _, sigma_oof_low, _ = forward_problem(f_true, K_hat_oof, 20)

    print('=== Part 1: Forward Problem ===')
    print(f'Gaussian     — sigma_noise (40 dB) = {sigma_high:.6f},  (20 dB) = {sigma_low:.6f}')
    print(f'Motion       — sigma_noise (40 dB) = {sigma_motion_high:.6f},  (20 dB) = {sigma_motion_low:.6f}')
    print(f'Out-of-Focus — sigma_noise (40 dB) = {sigma_oof_high:.6f},  (20 dB) = {sigma_oof_low:.6f}')
    print()

    # --- Penalty terms ---

    P_sq_L2 = ifftshift(np.ones((N, M)))
    P_sq_H1 = ifftshift(omega_X ** 2 + omega_Y ** 2)
    P_sq_H2 = ifftshift((omega_X ** 2 + omega_Y ** 2) ** 2)
    penalties = {'L2': P_sq_L2, 'H1': P_sq_H1, 'H2': P_sq_H2}

    # --- Tikhonov reconstruction ---

    mu_high_snr = 1e-3
    mu_low_snr = 1e-1

    print('=== Part 2: Tikhonov Reconstruction (Gaussian Blur) ===')
    for snr_label, g_hat, mu in [('40 dB', g_hat_high, mu_high_snr),
                                  ('20 dB', g_hat_low, mu_low_snr)]:
        print(f'  SNR = {snr_label}, mu = {mu}')
        for name, P_sq in penalties.items():
            recon = tikhonov_reconstruct(g_hat, K_hat_gauss, P_sq, mu)
            psnr = compute_psnr(f_true, recon)
            print(f'    {name}: PSNR = {psnr:.2f} dB')
    print()

    print('=== Part 2: Tikhonov Reconstruction (Motion Blur, 40 dB) ===')
    for name, P_sq in penalties.items():
        recon = tikhonov_reconstruct(g_hat_motion_high, K_hat_motion, P_sq, mu_high_snr)
        psnr = compute_psnr(f_true, recon)
        print(f'  {name}: PSNR = {psnr:.2f} dB')
    print()

    print('=== Part 2: Tikhonov Reconstruction (Out-of-Focus Blur, 40 dB) ===')
    for name, P_sq in penalties.items():
        recon = tikhonov_reconstruct(g_hat_oof_high, K_hat_oof, P_sq, mu_high_snr)
        psnr = compute_psnr(f_true, recon)
        print(f'  {name}: PSNR = {psnr:.2f} dB')
    print()

    # --- Spectral windowing ---

    omega_mag_fft = ifftshift(omega_mag)
    cutoffs = [10, 20, 40, 60, 80]

    print('=== Part 3: Spectral Windowing (Gaussian Blur, 40 dB) ===')
    best_Omega = None
    best_psnr_window = -np.inf
    for Omega in cutoffs:
        recon = spectral_window_reconstruct(g_hat_high, K_hat_gauss, Omega, omega_mag_fft)
        psnr = compute_psnr(f_true, recon)
        print(f'  Omega = {Omega}: PSNR = {psnr:.2f} dB')
        if psnr > best_psnr_window:
            best_psnr_window = psnr
            best_Omega = Omega

    print(f'  Best cutoff: Omega = {best_Omega} (PSNR = {best_psnr_window:.2f} dB)')
    print()

    # --- Bias-variance trade-off ---

    mu_range = np.logspace(-15, 0, 100)

    print('=== Part 4: Bias-Variance Trade-off ===')

    # H1 penalty at both SNR levels
    bias_h1, var_h1, total_h1 = compute_bias_variance(
        K_hat_gauss, P_sq_H1, f_true, sigma_high, mu_range)
    idx_opt_high = np.argmin(total_h1)
    mu_opt_high = mu_range[idx_opt_high]

    bias_h1_low, var_h1_low, total_h1_low = compute_bias_variance(
        K_hat_gauss, P_sq_H1, f_true, sigma_low, mu_range)
    idx_opt_low = np.argmin(total_h1_low)
    mu_opt_low = mu_range[idx_opt_low]

    print(f'  H1, SNR = 40 dB: optimal mu = {mu_opt_high:.4e}')
    print(f'    Bias^2   = {bias_h1[idx_opt_high]:.4e}')
    print(f'    Variance = {var_h1[idx_opt_high]:.4e}')
    print(f'    Total    = {total_h1[idx_opt_high]:.4e}')
    print()
    print(f'  H1, SNR = 20 dB: optimal mu = {mu_opt_low:.4e}')
    print(f'    Bias^2   = {bias_h1_low[idx_opt_low]:.4e}')
    print(f'    Variance = {var_h1_low[idx_opt_low]:.4e}')
    print(f'    Total    = {total_h1_low[idx_opt_low]:.4e}')
    print()

    # Reconstruction at optimal mu
    f_opt_high = tikhonov_reconstruct(g_hat_high, K_hat_gauss, P_sq_H1, mu_opt_high)
    f_opt_low = tikhonov_reconstruct(g_hat_low, K_hat_gauss, P_sq_H1, mu_opt_low)
    print(f'  Reconstruction at optimal mu:')
    print(f'    40 dB: PSNR = {compute_psnr(f_true, f_opt_high):.2f} dB')
    print(f'    20 dB: PSNR = {compute_psnr(f_true, f_opt_low):.2f} dB')
    print()

    # Total error comparison across all penalties
    print('  Total error comparison (L2 vs H1 vs H2, SNR = 40 dB):')
    for name, P_sq in penalties.items():
        _, _, total = compute_bias_variance(K_hat_gauss, P_sq, f_true, sigma_high, mu_range)
        idx = np.argmin(total)
        print(f'    {name}: optimal mu = {mu_range[idx]:.2e}, min total error = {total[idx]:.4e}')
