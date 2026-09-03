"""
Gaussian emission-line fitting for JWST NIRSpec PRISM spectra.
Fits Hβ, [OIII] 4959/5007, Hα, [NII] 6548/6583, [SII] 6716/6731
using emcee MCMC with curve_fit initialisation.
 
NII and SII are only fit if they are (a) covered without a chip gap,
(b) detected above a S/N threshold, and (c) resolvable given the
instrumental line spread function.
 
Usage (single galaxy):
    from gaussian_fitting import line_fitting, load_instrument_lsf
 
    R_interp = load_instrument_lsf("jwst_nirspec_prism_disp.fits")
    wave, flux, flux_err, df, fit_flags = line_fitting(
        wave_seg, flux_seg, err_seg,
        hb_center=HBETA_obs, oiii_center=OIII_5007_obs, ha_center=HALPHA_obs,
        R_interp=R_interp
    )

Usage (full sample, already-loaded spectra, one galaxy per core):
    from gaussian_fitting import run_line_fitting_batch, load_instrument_lsf

    R_interp = load_instrument_lsf("jwst_nirspec_prism_disp.fits")
    spectra_dict = {ID: (wave, flux, flux_err) for ID in sample_ids}
    results = run_line_fitting_batch(
        spectra_dict, R_interp, n_processes=8,
        hb_center=HBETA_obs, oiii_center=OIII_5007_obs, ha_center=HALPHA_obs,
    )

Usage (full sample, straight from a catalog CSV + Campfire spectra, one
galaxy per core):
    from gaussian_fitting import run_photspec_batch, load_instrument_lsf
    import pandas as pd

    R_interp = load_instrument_lsf("jwst_nirspec_prism_disp.fits")
    catalog = pd.read_csv("CEERS_photspec_1.csv")  # needs object_id, z_spec, ID_spec
    results_df = run_photspec_batch(
        catalog, R_interp,
        output_csv="CEERS_fitted.csv",
        plot_dir="./triple_line_plots/CEERS",
        n_processes=8,
    )
"""

import os
import traceback
from functools import partial
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; swap to TkAgg if working interactively
import matplotlib.pyplot as plt
 
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy.interpolate import interp1d, Akima1DInterpolator
from scipy.optimize import curve_fit
import emcee 
from multiprocessing import Pool

# Physical / model constants
OIII_RATIO = 2.98 # [OIII] 5007 / 4959  (fixed by atomic physics)
NII_RATIO = 2.94 # [NII]  6583 / 6548
SII_RATIO = 1.3 # [SII] 6716 / 6731, typical low-density HII region
 
# Minimum S/N for a doublet to be included as a free component
SNR_THRESHOLD_NII = 2.5
SNR_THRESHOLD_SII = 2.5
 
# A doublet is "resolvable" if the line separation exceeds this multiple of
# the instrumental sigma at that wavelength.
RESOLVABILITY_FACTOR = 1.0   # set to ~0.5–1.5 depending on how strict you want to be


def load_instrument_lsf(disp_file: str) -> interp1d:
    """
    Load the NIRSpec PRISM dispersion table and return an interpolator
    R(λ) where λ is in µm.
    """
    with fits.open(disp_file) as hdul:
        tab = hdul[1].data
        names = hdul[1].columns.names
        lam_key = [n for n in names if "wave" in n.lower()][0]
        R_key = next(n for n in names if "res" in n.lower() or n.lower() == "r")
        lam_R = tab[lam_key]
        R_vals = tab[R_key]
 
    return interp1d(lam_R, R_vals, bounds_error=False, fill_value="extrapolate")


def inst_sigma(lam, R_interp):
    """
    Gaussian instrumental sigma (µm) at wavelength lam (µm).
    The 0.7 factor accounts for the NIRSpec PRISM pixel sampling.
    """
    return lam / (2.355 * R_interp(lam)) * 0.7


def line_window(center, R_interp, n_sigma=8, min_window=0.003, max_window=0.05):
    """
    Adaptive fitting half-width for a single line: n_sigma instrumental
    sigmas at that wavelength, floored at min_window and capped at max_window 
    """
    return np.clip(n_sigma * inst_sigma(center, R_interp), min_window, max_window)


def gaussian(x, A, mu, sigma):
    """
    Gaussian Model for Line Fitting. This is of the form: Gaussian = Ae^(-(x-mu)^2/(2sigma^2))
    Returns: Evaluated Gaussian for the given A, mu and sigma at the point(s) in x
    """
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def full_line_model(x, A_hb, A_oiii, A_ha, mu_ha, R_nii, R_sii,
                      sigma_int, m, b,
                      R_interp=None,
                      fit_nii=True, fit_sii=True):
    """
    Full emission-line model:
        Hβ + [OIII]5007 + [OIII]4959 + Hα
        + [NII]6583/6548 (if fit_nii)
        + [SII]6716/6731 (if fit_sii)
        + linear continuum
 
    All line centres are tied to mu_ha via rest-frame wavelength ratios.
    sigma_int is the *intrinsic* velocity dispersion.
    Observed σ = sqrt(sigma_int^2 + sigma_inst^2).
    """

    if R_interp is None:
        raise ValueError("R_interp must be provided to full_line_model")
 
    mu_hb = mu_ha * 4861 / 6563
    mu_oiii_5007 = mu_ha * 5007 / 6563
    mu_oiii_4959 = mu_ha * 4959 / 6563
    mu_nii_6583 = mu_ha * 6583 / 6563
    mu_nii_6548 = mu_ha * 6548 / 6563
    mu_sii_6716 = mu_ha * 6716 / 6563
    mu_sii_6731 = mu_ha * 6731 / 6563
 
    all_mus  = np.array([mu_hb, mu_oiii_5007, mu_oiii_4959, 
                        mu_nii_6583, mu_nii_6548, mu_sii_6716, mu_sii_6731, mu_ha])
    all_inst = inst_sigma(all_mus, R_interp)
 
    def sigma_tot(inst):
        return np.sqrt(sigma_int ** 2 + inst ** 2)
 
    (sig_hb, sig_oiii, sig_oiii_4959, sig_nii_6583, sig_nii_6548,
     sig_sii_6716, sig_sii_6731, sig_ha) = sigma_tot(all_inst)
 
    A_nii = (R_nii * A_ha) if fit_nii else 0.0
    A_sii = (R_sii * A_ha) if fit_sii else 0.0
 
    model = (
        gaussian(x, A_hb, mu_hb, sig_hb)
        + gaussian(x, A_oiii, mu_oiii_5007, sig_oiii)
        + gaussian(x, A_oiii / OIII_RATIO, mu_oiii_4959, sig_oiii_4959)
        + gaussian(x, A_ha, mu_ha, sig_ha)
        + gaussian(x, A_nii, mu_nii_6583, sig_nii_6583)
        + gaussian(x, A_nii / NII_RATIO, mu_nii_6548, sig_nii_6548) 
        + gaussian(x, A_sii, mu_sii_6716, sig_sii_6716)
        + gaussian(x, A_sii / SII_RATIO, mu_sii_6731, sig_sii_6731)
        + m * x + b
    )
    return model


def make_model_wrapper(R_interp, fit_nii, fit_sii):
    """
    Return a curve_fit-compatible wrapper around full_line_model
    with R_interp, fit_nii, fit_sii fixed via closure.
    """
    def wrapper(x, A_hb, A_oiii, A_ha, mu_ha, R_nii, R_sii,
                sigma_int, m, b):
        return full_line_model(x, A_hb, A_oiii, A_ha, mu_ha, R_nii, R_sii,
                                 sigma_int, m, b, R_interp=R_interp, 
                                 fit_nii=fit_nii, fit_sii=fit_sii)
    return wrapper


def detect_line_snr(wave, spec, err, center, R_interp, n_sigma_window=3):
    """
    Estimate the signal-to-noise ratio for a single emission line.
 
    Returns:
    snr (float)
        Peak S/N.  Returns 0.0 if the window contains no data.
    """
    sig = inst_sigma(center, R_interp)
    hw = n_sigma_window * sig
 
    # if less than 2 pixels in the line window, return 0.0 to avoid unreliable estimates
    line_mask = (wave > center - hw) & (wave < center + hw)
    if line_mask.sum() < 2:
        return 0.0
 
    # define sidebands for noise estimation: 3–6 sigma on either side of the line center
    sb_mask = (
        ((wave > center - 6 * sig) & (wave < center - 3 * sig)) |
        ((wave > center + 3 * sig) & (wave < center + 6 * sig))
    )
    if sb_mask.sum() < 3:
        # Fall back to a local RMS from the error array
        noise = np.nanmedian(err[line_mask])
    else:
        noise = np.nanstd(spec[sb_mask])
 
    if not np.isfinite(noise) or noise <= 0:
        noise = np.nanmedian(err[line_mask])
    if not np.isfinite(noise) or noise <= 0:
        return 0.0
 
    # Estimate the continuum level as the median in the sidebands, then compute peak S/N
    cont_level = np.nanmedian(spec[sb_mask]) if sb_mask.sum() >= 3 else 0.0
    peak = np.nanmax(spec[line_mask]) - cont_level
 
    return float(peak / noise)

def check_line_covered(wave, center, R_interp, n_sigma=3):
    """
    Return True if there are spectral pixels within ±n_sigma * sigma_inst of center.
    """
    hw = n_sigma * inst_sigma(center, R_interp)
    nearby = np.sum((wave > center - hw) & (wave < center + hw))
    return nearby > 0


def check_sii_resolvable(sii_6716_center, sii_6731_center, R_interp):
    """
    Return True if the SII doublet separation exceeds RESOLVABILITY_FACTOR
    times the instrumental sigma at the doublet midpoint.
    """
    midpoint  = 0.5 * (sii_6716_center + sii_6731_center)
    sep = sii_6731_center - sii_6716_center
    sig = inst_sigma(midpoint, R_interp)
    return sep > RESOLVABILITY_FACTOR * sig


def log_likelihood(theta, x, y, yerr, R_interp, fit_nii, fit_sii):
    model = full_line_model(x, *theta, R_interp=R_interp, fit_nii=fit_nii, fit_sii=fit_sii)
    # Reject non-finite model 
    if not np.all(np.isfinite(model)):
        return -np.inf
    # Reject non-positive or non-finite errors
    if np.any(yerr <= 0) or not np.all(np.isfinite(yerr)):
        return -np.inf

    lnL = -0.5 * np.sum(((y - model) / yerr) ** 2 + np.log(2 * np.pi * yerr ** 2))
    return lnL if np.isfinite(lnL) else -np.inf


def log_prior(theta, ha_center, amp_max_dict, sigma_int_max, delta_mu,
              m_bound, b_range, sigma_inst_at_ha, fit_nii, fit_sii):
    A_hb, A_oiii, A_ha, mu_ha, R_nii, R_sii, sigma_int, m, b = theta
 
    if not np.all(np.isfinite(theta)):
        return -np.inf

    if not (0 < A_hb < amp_max_dict['hb']): return -np.inf
    if not (0 < A_oiii < amp_max_dict['oiii']): return -np.inf
    if not (0 < A_ha < amp_max_dict['ha']): return -np.inf
    R_nii_hi = 3.0 if fit_nii else 1e-6
    R_sii_hi = 2.0 if fit_sii else 1e-6
    if not (0 <= R_nii <= R_nii_hi): return -np.inf
    if not (0 <= R_sii <= R_sii_hi): return -np.inf
 
    sigma_int_lo = sigma_inst_at_ha * 0.1
    if not (sigma_int_lo < sigma_int < sigma_int_max): return -np.inf
    # Centroid must be within delta_mu of the expected Halpha position
    if not (ha_center - delta_mu < mu_ha < ha_center + delta_mu): return -np.inf
    # Constrain slope and intercept to reasonable ranges
    if not (-m_bound < m < m_bound): return -np.inf
    if not (-b_range < b < b_range): return -np.inf
 
    return 0.0


def log_probability(theta, x, y, yerr, ha_center, amp_max_dict, sigma_int_max, delta_mu,
                    m_bound, b_range, sigma_inst_at_ha, R_interp, fit_nii, fit_sii):
    lp = log_prior(theta, ha_center, amp_max_dict, sigma_int_max, delta_mu,
              m_bound, b_range, sigma_inst_at_ha, fit_nii, fit_sii)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta, x, y, yerr, R_interp, fit_nii, fit_sii)


def initial_fits(wave, spectrum, err_spec, window, hb_center, oiii_center, ha_center,
                 R_interp, fit_nii, fit_sii, diagnose=False,
                 window_n_sigma=8, window_max=0.05):
    """
    Use curve_fit to get a starting point for MCMC.
 
    Parameters:
    fit_nii, fit_sii : bool
        Whether NII / SII are free components.  If False the corresponding
        ratio parameters are pinned near zero via bounds.
    window_n_sigma, window_max : float
        Passed straight through to line_window() for the adaptive
        per-line fitting mask (see line_window docstring).
 
    Returns:
    popt, delta_mu, m_bound, b_range, sigma_int_hi
    """
    sii_6716_center = ha_center * 6716 / 6563
    sii_6731_center = ha_center * 6731 / 6563
 
    # build fitting window; everything outside of cutouts is ignored in the fit
    w_hb = line_window(hb_center, R_interp, n_sigma=window_n_sigma, max_window=window_max)
    w_oiii = line_window(oiii_center, R_interp, n_sigma=window_n_sigma, max_window=window_max)
    w_ha = line_window(ha_center, R_interp, n_sigma=window_n_sigma, max_window=window_max)
    w_sii1 = line_window(sii_6716_center, R_interp, n_sigma=window_n_sigma, max_window=window_max)
    w_sii2 = line_window(sii_6731_center, R_interp, n_sigma=window_n_sigma, max_window=window_max)

    mask = (
        ((wave > hb_center - w_hb) & (wave < hb_center + w_hb)) |
        ((wave > oiii_center - w_oiii) & (wave < oiii_center + w_oiii)) |
        ((wave > ha_center - w_ha) & (wave < ha_center + w_ha)) |
        ((wave > sii_6716_center - w_sii1) & (wave < sii_6716_center + w_sii1)) |
        ((wave > sii_6731_center - w_sii2) & (wave < sii_6731_center + w_sii2))
    )

    wave_window = wave[mask]
    spec_window = spectrum[mask]
    err_window = err_spec[mask]

    if len(wave_window) < 10:
        raise ValueError("Too few pixels in fitting window.")
 
    # Mandatory coverage check (Hbeta, [OIII], Halpha only) 
    for center, label in [(hb_center, 'Hβ'), (oiii_center, '[OIII]'), (ha_center, 'Hα')]:
        if not check_line_covered(wave_window, center, R_interp):
            raise ValueError(f"Chip gap or missing coverage at {label} ({center:.4f} µm)")
 
    # Continuum estimate (removes line pixels and outliers to avoid biasing the fit)
    line_mask = (
        ((wave > hb_center - inst_sigma(hb_center, R_interp) * 3)
         & (wave < hb_center + inst_sigma(hb_center, R_interp) * 3)) |
        ((wave > oiii_center - inst_sigma(oiii_center, R_interp) * 3)
         & (wave < oiii_center + inst_sigma(oiii_center, R_interp) * 3)) |
        ((wave > ha_center - inst_sigma(ha_center, R_interp) * 3)
         & (wave < ha_center + inst_sigma(ha_center, R_interp) * 3)) |
        ((wave > sii_6716_center - inst_sigma(sii_6716_center, R_interp) * 3)
         & (wave < sii_6716_center + inst_sigma(sii_6716_center, R_interp) * 3)) |
        ((wave > sii_6731_center - inst_sigma(sii_6731_center, R_interp) * 3)
         & (wave < sii_6731_center + inst_sigma(sii_6731_center, R_interp) * 3))
    )
    cont_mask = (wave > hb_center - window) & (wave < sii_6731_center + window)
    cw_all = wave[cont_mask & ~line_mask]
    cs_all = spectrum[cont_mask & ~line_mask]

    # fits straight line to the continuum pixels; if too few pixels remain after masking, fall back to a flat continuum at the median level
    if len(cw_all) < 4:
        guess_m, guess_b = 0.0, np.nanmedian(spec_window)
    else:
        clipped = sigma_clip(cs_all, sigma=2.5, maxiters=5)
        cw = cw_all[~clipped.mask]
        cs = cs_all[~clipped.mask]
        if len(cw) < 4:
            cw, cs = cw_all, cs_all
        guess_m, guess_b = np.polyfit(cw, cs, 1)
    
    # Set bounds for slope and intercept based on continuum variability and noise level
    spec_rms = np.nanstd(spec_window)
    m_bound = max(abs(guess_m) * 3, spec_rms / (wave_window[-1] - wave_window[0]) * 5)
    b_range = max(abs(guess_b) * 3, spec_rms * 5, np.nanmax(np.abs(spec_window)) * 1.5)
 
    # Guess sigma from Halpha FWHM 
    sigma_inst_at_ha = inst_sigma(ha_center, R_interp)
    ha_hw = 5 * sigma_inst_at_ha
    ha_mask = (wave_window > ha_center - ha_hw) & (wave_window < ha_center + ha_hw)
    wave_ha = wave_window[ha_mask]
    spec_ha = spec_window[ha_mask]
 
    def safe_sigma(val, lo=1e-9, hi=0.02):
        return np.clip(val if np.isfinite(val) else 0.003, lo, hi)
 
    try:
        interp_ha = Akima1DInterpolator(wave_ha, spec_ha)
        x_fine = np.linspace(wave_ha.min(), wave_ha.max(), 5000)
        y_fine = interp_ha(x_fine)
        cont_at_ha = guess_m * ha_center + guess_b
        half_max = (np.nanmax(y_fine) - cont_at_ha) / 2 + cont_at_ha
        idx = np.where(y_fine > half_max)[0]
        if len(idx) < 2:
            raise ValueError("cannot measure FWHM")
        fwhm = x_fine[idx[-1]] - x_fine[idx[0]]
        guess_sigma_ha = fwhm / 2.355
        if guess_sigma_ha < sigma_inst_at_ha * 0.8:
            raise ValueError("FWHM unphysically narrow")
    except Exception:
        guess_sigma_ha = sigma_inst_at_ha
 
    guess_sigma_tot = safe_sigma(max(guess_sigma_ha, sigma_inst_at_ha))
    guess_sigma_int = np.sqrt(max(guess_sigma_tot ** 2 - sigma_inst_at_ha ** 2, 1e-12))
    sigma_int_lo = sigma_inst_at_ha * 0.1
    sigma_int_hi = ha_center * (500 / 3e5) 
 
    def cont_at(lam):
        return guess_m * lam + guess_b

    # Floor for degenerate amplitude guesses, scaled to the actual noise level
    # of this spectrum rather than a fixed absolute number.
    _err_med = np.nanmedian(err_window)
    amp_floor = 1e-3 * _err_med if (np.isfinite(_err_med) and _err_med > 0) else 1e-6

    # subtracts continuum and averages pixels to get a rough line amplitude guess
    def line_amp_guess(center):
        hw  = 2 * inst_sigma(center, R_interp)
        lmask = (wave_window > center - hw) & (wave_window < center + hw)
        if lmask.sum() == 0:
            hw  = window / 4
            lmask = (wave_window > center - hw) & (wave_window < center + hw)
        if lmask.sum() == 0:
            return amp_floor
        w = np.exp(-0.5 * ((wave_window[lmask] - center) / inst_sigma(center, R_interp)) ** 2)
        if w.sum() == 0:
            return amp_floor
        val = np.average(spec_window[lmask] - cont_at(wave_window[lmask]), weights=w)
        return max(float(val), amp_floor)
 
    guess_A_ha = line_amp_guess(ha_center)
    guess_A_oiii = line_amp_guess(oiii_center)
    guess_A_hb = line_amp_guess(hb_center)
    sii_amp = line_amp_guess(sii_6716_center)
 
    guess_R_nii = 0.2
    guess_R_sii = np.clip(sii_amp / max(guess_A_ha, 1e-6), 0.0, 2.0)
 
    delta_mu = 3 * sigma_inst_at_ha
    guess_mu_ha = ha_center
 
    # Bounds depend on fit_nii / fit_sii flags 
    R_nii_lo, R_nii_hi = (0.0, 3.0) if fit_nii else (0.0, 1e-6)
    R_sii_lo, R_sii_hi = (0.0, 2.0) if fit_sii else (0.0, 1e-6)
 
    # Zero out guesses for unfitted components so curve_fit doesn't wander
    if not fit_nii:
        guess_R_nii = 0.0
    if not fit_sii:
        guess_R_sii = 0.0

    spec_max = np.nanmax(spec_window)
 
    # initial parameter vector and bounds for curve_fit
    p0 = [
        guess_A_hb, guess_A_oiii,
        guess_A_ha, guess_mu_ha,
        guess_R_nii, guess_R_sii,
        guess_sigma_int,
        guess_m, guess_b,
    ]
    low_bounds = [
        0, 0,
        0, ha_center - delta_mu,
        R_nii_lo, R_sii_lo,
        sigma_int_lo,
        -m_bound, -b_range,
    ]
    high_bounds = [
        max(2 * guess_A_hb, spec_max),
        max(2 * guess_A_oiii, spec_max),
        max(2 * guess_A_ha, spec_max), ha_center + delta_mu,
        R_nii_hi, R_sii_hi,
        sigma_int_hi,
        m_bound, b_range,
    ]
 
    # Clamp p0 inside bounds with a small buffer, scaled to each parameter's
    # own bound span rather than a fixed absolute epsilon. 
    low_arr = np.array(low_bounds, dtype=float)
    high_arr = np.array(high_bounds, dtype=float)
    span = high_arr - low_arr
    if np.any(span <= 0):
        bad = np.where(span <= 0)[0]
        names = ['A_hb', 'A_oiii', 'A_ha', 'mu_ha', 'R_nii', 'R_sii', 'sigma_int', 'm', 'b']
        details = ", ".join(
            f"{names[i]} (low={low_arr[i]:.3g}, high={high_arr[i]:.3g})" for i in bad
        )
        raise ValueError(f"Degenerate parameter bounds: {details}")
    eps = span * 1e-6
    p0 = np.clip(p0, low_arr + eps, high_arr - eps)
 
    model_fn = make_model_wrapper(R_interp, fit_nii, fit_sii)

    # Try the standard guess first, then a few alternate sigma seeds if it fails.
    sigma_seed_multipliers = [1.0, 2.0, 0.5, 4.0]
    popt = None
    converged = False
    best_resid = np.inf

    for mult in sigma_seed_multipliers:
        p0_try = np.array(p0, dtype=float)
        p0_try[6] = np.clip(guess_sigma_int * mult, low_bounds[6] + 1e-10, high_bounds[6] - 1e-10)
        try:
            popt_try, _ = curve_fit(
                model_fn, wave_window, spec_window,
                p0=p0_try, sigma=err_window,
                bounds=(low_bounds, high_bounds),
                maxfev=30_000,
            )
            resid = np.sum(((spec_window - model_fn(wave_window, *popt_try)) / err_window) ** 2)
            if resid < best_resid:
                best_resid = resid
                popt = popt_try
                converged = True
        except RuntimeError:
            continue

    if popt is None:
        print("curve_fit did not converge on any seed; using p0 as fallback.")
        popt = np.array(p0)
        converged = False
 
    if diagnose:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(wave_window, spec_window, 'k.', ms=3, label='Data')
        x_fit = np.linspace(wave_window[0], wave_window[-1], 800)
        ax.plot(x_fit, model_fn(x_fit, *popt), 'r-', lw=1.5, label='Initial fit')
        for lam, lbl, col in [
            (hb_center, 'Hβ', 'steelblue'),
            (oiii_center, '[OIII]', 'seagreen'),
            (ha_center, 'Hα', 'mediumpurple'),
            (sii_6716_center,'[SII]6716','darkorange'),
            (sii_6731_center,'[SII]6731','darkorange'),
        ]:
            ax.axvline(lam, color=col, ls='--', lw=0.8, label=lbl)
        ax.legend(fontsize=7, ncol=3)
        ax.set_xlabel('Wavelength (µm)')
        plt.tight_layout()
        plt.show()
 
    amp_scales = {'hb': popt[0], 'oiii': popt[1], 'ha': popt[2]}

    return popt, delta_mu, m_bound, b_range, sigma_int_hi, amp_scales, converged


def line_fitting(wave, flux, flux_err, R_interp, hb_center=0.4867, oiii_center=0.5007, ha_center=0.6563,
                   window=0.01, snr_thresh_nii=SNR_THRESHOLD_NII, snr_thresh_sii=SNR_THRESHOLD_SII,
                   nwalkers=32, steps=5000, burnin=3000, diagnose=False, window_n_sigma=8, window_max=0.05):
    """
    Fit emission lines with MCMC and return posterior samples.
 
    NII and SII are only included as free parameters if they pass S/N and resolvability checks.
 
    Parameters:
    wave, flux, flux_err : array-like
        Observed wavelength (µm), flux, and 1-sigma flux uncertainty.
    R_interp : callable
        Spectral resolution interpolator from load_instrument_lsf().
    hb_center, oiii_center, ha_center : float
        Observed-frame line centres in µm.
    window : float
        Half-width (µm) used for the continuum mask and amplitude-guess
        fallback. Does NOT set the per-line fitting mask (see
        window_n_sigma / window_max below).
    window_n_sigma, window_max : float
        Control the adaptive per-line fitting window (line_window()):
        n_sigma instrumental sigmas, capped at window_max. 
    snr_thresh_nii, snr_thresh_sii : float
        Minimum peak S/N required to include NII / SII as free components.
    nwalkers, steps, burnin : int
        MCMC sampler settings.
    diagnose : bool
        If True, show diagnostic plots.
 
    Returns
    -------
    wave, flux, flux_err : arrays (pass-through)
    df : pd.DataFrame
        Posterior samples with derived fluxes.
    fit_flags : dict
        {'fit_nii': bool, 'fit_sii': bool,
         'snr_nii': float, 'snr_sii': float, 'sii_resolvable': bool}
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
 
    sii_6716_center = ha_center * 6716 / 6563
    sii_6731_center = ha_center * 6731 / 6563
    nii_6583_center = ha_center * 6583 / 6563
 
    # Pre-fit detection / resolvability checks for optional doublets
    snr_nii = detect_line_snr(wave, flux, flux_err, nii_6583_center, R_interp)
    snr_sii = detect_line_snr(wave, flux, flux_err, sii_6716_center,  R_interp)
    sii_res = check_sii_resolvable(sii_6716_center, sii_6731_center, R_interp)
 
    # Coverage check: skip optional doublets that fall in chip gaps
    nii_covered = check_line_covered(wave, nii_6583_center,  R_interp)
    sii_covered = (check_line_covered(wave, sii_6716_center, R_interp) and
                   check_line_covered(wave, sii_6731_center, R_interp))
 
    fit_nii = nii_covered and (snr_nii >= snr_thresh_nii)
    fit_sii = sii_covered and (snr_sii >= snr_thresh_sii) and sii_res
 
    fit_flags = {
        'fit_nii': fit_nii,
        'fit_sii': fit_sii,
        'snr_nii': snr_nii,
        'snr_sii': snr_sii,
        'sii_resolvable': sii_res,
    }
    print(f"NII: SNR={snr_nii:.1f}, fit={fit_nii} | "
          f"SII: SNR={snr_sii:.1f}, resolvable={sii_res}, fit={fit_sii}")
 
    # Initial parameter estimation
    p0, delta_mu, m_bound, b_range, sigma_int_hi, amp_scales, curve_fit_converged = initial_fits(
        wave, flux, flux_err, window, hb_center, oiii_center, ha_center,
        R_interp, fit_nii, fit_sii, diagnose=diagnose,
        window_n_sigma=window_n_sigma, window_max=window_max,
    )
 
    # MCMC initialisation
    ndim = 9
    sigma_inst_at_ha = inst_sigma(p0[3], R_interp)
    sigma_int_lo = sigma_inst_at_ha * 0.1
 
    # for amplitude parameters that must be positive
    def pos_walkers(center, scale, n):
        return np.clip(np.random.normal(center, scale, n), 1e-6, None).reshape(-1, 1)

    # for parameters with hard bounds
    def bnd_walkers(center, scale, lo, hi, n):
        return np.clip(np.random.normal(center, scale, n), lo, hi).reshape(-1, 1)
 
    # for ratio parameters, ensure initial walkers are not too close to the bounds to avoid getting stuck
    def safe_p0(val, lo, hi, buf_frac=1e-4):
        buf = (hi - lo) * buf_frac
        return np.clip(val, lo + buf, hi - buf)
 
    # Ratio bounds mirror initial_fits
    R_nii_lo, R_nii_hi = (0.0, 3.0) if fit_nii else (0.0, 1e-6)
    R_sii_lo, R_sii_hi = (0.0, 2.0) if fit_sii else (0.0, 1e-6)
 
    # builds the full walker
    pos = np.hstack([
        pos_walkers(p0[0], max(p0[0] / 10, 1e-6), nwalkers),
        pos_walkers(p0[1], max(p0[1] / 10, 1e-6), nwalkers),
        pos_walkers(p0[2], max(p0[2] / 10, 1e-6), nwalkers),
        bnd_walkers(p0[3], delta_mu / 3, ha_center - delta_mu, ha_center + delta_mu, nwalkers),
        bnd_walkers(safe_p0(p0[4], R_nii_lo, R_nii_hi), 0.05, R_nii_lo, R_nii_hi, nwalkers),
        bnd_walkers(safe_p0(p0[5], R_sii_lo, R_sii_hi), 0.05, R_sii_lo, R_sii_hi, nwalkers),
        bnd_walkers(safe_p0(p0[6], sigma_int_lo, sigma_int_hi),
                    sigma_inst_at_ha * 0.2, sigma_int_lo, sigma_int_hi, nwalkers),
        bnd_walkers(p0[7], max(abs(p0[7]) / 10, m_bound / 10), -m_bound, m_bound, nwalkers),
        bnd_walkers(p0[8], max(abs(p0[8]) / 10, b_range / 10), -b_range, b_range, nwalkers),
    ])
    pos += np.random.normal(0, 1e-10, pos.shape)   # break exact degeneracies
 
    spec_max = float(np.nanmax(flux))
    amp_max = {
        'hb': max(5 * amp_scales['hb'], spec_max, 1e-6),
        'oiii': max(5 * amp_scales['oiii'], spec_max, 1e-6),
        'ha': max(5 * amp_scales['ha'], spec_max, 1e-6),
    }

    sampler_args = (wave, flux, flux_err,
                    ha_center, amp_max, sigma_int_hi, delta_mu,
                    m_bound, b_range, sigma_inst_at_ha,
                    R_interp, fit_nii, fit_sii)

    # Reject-and-resample any walker that starts at -inf log-probability.
    n_stuck = 0
    p0_arr = np.asarray(p0, dtype=float)
    for i in range(nwalkers):
        tries = 0
        lp = log_probability(pos[i], *sampler_args)
        while not np.isfinite(lp) and tries < 50:
            scale = 0.05 * (1 + tries // 10)   # widen jitter every 10 failed tries
            candidate = p0_arr * (1 + np.random.normal(0, scale, size=p0_arr.shape))
            candidate[0:3] = np.abs(candidate[0:3])   # amplitudes must stay positive
            pos[i] = candidate
            lp = log_probability(pos[i], *sampler_args)
            tries += 1
        if not np.isfinite(lp):
            pos[i] = p0_arr.copy()   # last resort: exact known-good p0
            lp = log_probability(pos[i], *sampler_args)
        if not np.isfinite(lp):
            n_stuck += 1
    if n_stuck > 0:
        print(f"  WARNING: {n_stuck}/{nwalkers} walkers still stuck at -inf even at p0 -- check this galaxy's data")

 
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability,
        args=sampler_args,
    )
    sampler.run_mcmc(pos, steps, progress=True)

    # Check acceptance fraction — should be 0.2–0.5
    acc_frac = np.mean(sampler.acceptance_fraction)
    print(f"  Mean acceptance fraction: {acc_frac:.3f}")
    if not (0.1 < acc_frac < 0.7):
        print(f"  WARNING: acceptance fraction {acc_frac:.3f} outside healthy range")

    # Check autocorrelation time
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        max_tau = np.nanmax(tau)
        if np.isnan(max_tau):
            effective_steps = 0.0
        else:
            effective_steps = (steps - burnin) / max_tau
        print(f"  Max autocorr time: {max_tau:.1f}, "
            f"effective samples per walker: {effective_steps:.1f}")
    except emcee.autocorr.AutocorrError:
        effective_steps = 0.0
        print("  WARNING: autocorrelation time could not be estimated")

    fit_flags['effective_steps'] = float(effective_steps)
    fit_flags['acc_frac'] = float(acc_frac)
    fit_flags['curve_fit_converged'] = bool(curve_fit_converged)

    fit_flags['reliable'] = bool(
        (0.05 < fit_flags['acc_frac'] < 0.9)
        and (fit_flags['effective_steps'] >= 3)
    )
 
    # Posterior processing
    flat = sampler.get_chain(discard=burnin, flat=True)
    lnL = sampler.get_log_prob(discard=burnin, flat=True)

    df = pd.DataFrame(flat, columns=[
        'A_hb', 'A_oiii', 'A_ha', 'mu_ha', 'R_nii', 'R_sii', 'sigma_int', 'm', 'b',
    ])
    df['LnL'] = lnL
    df = df[np.isfinite(df.LnL)].copy()
 
    # Derived line centres from posterior mu_ha
    mu_hb_s = df['mu_ha'] * 4861 / 6563
    mu_oiii_s = df['mu_ha'] * 5007 / 6563
    mu_sii_6716_s = df['mu_ha'] * 6716 / 6563
    mu_sii_6731_s = df['mu_ha'] * 6731 / 6563
 
    df['sigma_ha_broad'] = np.sqrt(np.maximum(df['sigma_int'] ** 2 + inst_sigma(df['mu_ha'], R_interp) ** 2, 1e-20))
    df['sigma_hb_broad'] = np.sqrt(np.maximum(df['sigma_int'] ** 2 + inst_sigma(mu_hb_s, R_interp) ** 2, 1e-20))
    df['sigma_oiii_broad'] = np.sqrt(np.maximum(df['sigma_int'] ** 2 + inst_sigma(mu_oiii_s, R_interp) ** 2, 1e-20))
    df['sigma_sii_6716_broad'] = np.sqrt(np.maximum(df['sigma_int']**2 + inst_sigma(mu_sii_6716_s, R_interp)**2, 1e-20))
    df['sigma_sii_6731_broad'] = np.sqrt(np.maximum(df['sigma_int']**2 + inst_sigma(mu_sii_6731_s, R_interp)**2, 1e-20))
    
    df['A_sii'] = df['R_sii'] * df['A_ha']
    df['A_nii'] = df['R_nii'] * df['A_ha']
 
    # converts amplitudes and sigmas to fluxes using the Gaussian integral
    df['Flux_Ha'] = df['A_ha'] * df['sigma_ha_broad'] * np.sqrt(2 * np.pi)
    df['Flux_Hb'] = df['A_hb'] * df['sigma_hb_broad'] * np.sqrt(2 * np.pi)
    df['Flux_OIII'] = df['A_oiii'] * df['sigma_oiii_broad'] * np.sqrt(2 * np.pi)
    df['Flux_SII_6716'] = df['A_sii'] * df['sigma_sii_6716_broad'] * np.sqrt(2 * np.pi)
    df['Flux_SII_6731'] = (df['A_sii'] / SII_RATIO) * df['sigma_sii_6731_broad'] * np.sqrt(2 * np.pi)
    df['Flux_SII'] = df['Flux_SII_6716'] + df['Flux_SII_6731']
 
    if diagnose:
        df.hist(figsize=(14, 10), bins=50)
        plt.tight_layout()
        plt.show()
 
        xarr = np.linspace(hb_center - 0.03, sii_6731_center + 0.03, 1200)
        med  = df.quantile(0.5)[['A_hb', 'A_oiii', 'A_ha', 'mu_ha',
                                  'R_nii', 'R_sii', 'sigma_int', 'm', 'b']].values
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.step(wave, flux, where='mid', color='black', alpha=0.5, label='Data')
        ax.fill_between(wave, flux - flux_err, flux + flux_err, alpha=0.2, color='grey')
        ax.plot(xarr, 
                full_line_model(xarr, *med, R_interp=R_interp, fit_nii=fit_nii, fit_sii=fit_sii),
                color='royalblue', lw=1.8, label='Median model')
        for lam, lbl, col in [
            (hb_center, 'Hβ', 'steelblue'),
            (oiii_center, '[OIII]', 'seagreen'),
            (ha_center, 'Hα', 'mediumpurple'),
            (sii_6716_center,'[SII]6716','darkorange'),
            (sii_6731_center,'[SII]6731','darkorange'),
        ]:
            ax.axvline(lam, color=col, ls='--', lw=0.8, label=lbl)
        ax.legend(fontsize=7, ncol=4)
        ax.set_xlabel('Wavelength (µm)')
        plt.tight_layout()
        plt.show()
 
    return wave, flux, flux_err, df, fit_flags


def posterior_summary(df):
    """
    Return a dict of median ± half 16th–84th percentile interval
    for the four main line fluxes and their S/N.
    """
    out = {}
    for col in ('Flux_Ha', 'Flux_Hb', 'Flux_OIII', 'Flux_SII'):
        med = np.median(df[col])
        lerr = (np.percentile(df[col], 84) - np.percentile(df[col], 16)) / 2
        out[col] = med
        out[col + '_err'] = lerr
        out[col + '_snr'] = med / lerr if lerr > 0 else np.nan
        out[col + '_p16'] = np.percentile(df[col], 16)
        out[col + '_p84'] = np.percentile(df[col], 84)
    return out


def _fit_one_spectrum(item, R_interp, line_fit_kwargs):
    """Worker for run_line_fitting_batch: fits one already-loaded spectrum."""
    ID, (wave, flux, flux_err) = item
    try:
        out = line_fitting(wave, flux, flux_err, R_interp, **line_fit_kwargs)
        return ID, out, None
    except Exception as e:
        return ID, None, f"{e}\n{traceback.format_exc()}"


def run_line_fitting_batch(spectra_dict, R_interp, n_processes=8, **line_fit_kwargs):
    """
    Run line_fitting() over a full sample of already-loaded spectra,
    fitting up to n_processes galaxies concurrently -- one per core.
 
    Parameters
    ----------
    spectra_dict : dict
        {ID: (wave, flux, flux_err)} for every galaxy to fit.
    R_interp : callable
        Spectral resolution interpolator from load_instrument_lsf().
    n_processes : int
        Number of worker processes.
    **line_fit_kwargs :
        Passed straight through to line_fitting() (e.g. hb_center,
        oiii_center, ha_center, window, steps, burnin, nwalkers, diagnose).
 
    Returns
    -------
    results : dict
        {ID: (wave, flux, flux_err, df, fit_flags)} on success,
        {ID: None} for any galaxy that raised an exception.
    """
    results = {}
    worker = partial(_fit_one_spectrum, R_interp=R_interp, line_fit_kwargs=line_fit_kwargs)
    with Pool(processes=n_processes) as pool:
        for ID, out, err in pool.imap_unordered(worker, spectra_dict.items()):
            if err is None:
                print(f"--- Fitting {ID}: done ---")
                results[ID] = out
            else:
                print(f"--- Fitting {ID}: FAILED ---\n  → {err}")
                results[ID] = None
    return results


_WORKER_CAMPFIRE = None
_WORKER_CAMPFIRE_DATA_DIR = None


def _init_campfire_worker(data_dir=None):
    """
    Pool initializer: create one Campfire client per worker process.

    data_dir, if given, is passed straight to Campfire(data_dir=...),
    overriding whatever $CAMPFIRE_ROOT happens to resolve to in this
    process's environment.
    """
    global _WORKER_CAMPFIRE, _WORKER_CAMPFIRE_DATA_DIR
    from campfire import Campfire
    _WORKER_CAMPFIRE_DATA_DIR = data_dir
    _WORKER_CAMPFIRE = Campfire(data_dir=data_dir) if data_dir else Campfire()


def _get_worker_campfire():
    """Return this worker's Campfire client, creating it on first use."""
    global _WORKER_CAMPFIRE
    if _WORKER_CAMPFIRE is None:
        _init_campfire_worker(_WORKER_CAMPFIRE_DATA_DIR)
    return _WORKER_CAMPFIRE


def _process_photspec_row(idx_row, R_interp, window, window_n_sigma, window_max,
                            plot_dir, line_fit_kwargs, grating='PRISM'):
    """
    Worker for run_photspec_batch: pulls one galaxy's spectrum via Campfire
    (obj.spectra -> .open()), runs the full fitting + summary +
    diagnostic-plot pipeline, and returns a plain dict of results (nothing
    containing open file handles / plots, so it pickles cleanly back to
    the parent process).
    """
    idx, row = idx_row
    object_id = row["object_id"]
    z = row["z_spec"]
    spec_id = row["ID_spec"]

    result = {'idx': idx, 'spec_id': spec_id, 'status': 'ok', 'message': None, 'data': None}

    HBETA_obs = 0.4861 * (1 + z)
    OIII_obs = 0.5007 * (1 + z)
    OIII_4959 = 0.4959 * (1 + z)
    HALPHA_obs = 0.6563 * (1 + z)
    SII_6716 = 0.6716 * (1 + z)
    SII_6731 = 0.6731 * (1 + z)
    NII_6548 = 0.6548 * (1 + z)
    NII_6583 = 0.6583 * (1 + z)

    try:
        cf = _get_worker_campfire()
        obj = cf.get_object(object_id)
        candidates = obj.spectra[obj.spectra.grating == grating]
        if len(candidates) == 0:
            result['status'] = 'skip'
            result['message'] = f"No {grating} spectrum found for {object_id}."
            return result
        if len(candidates) > 1:
            # More than one PRISM spectrum (e.g. multiple programs/visits) --
            # take the highest S/N one.
            spec_row = candidates[np.argmax(candidates.signal_to_noise)]
        else:
            spec_row = candidates[0]
        spec = cf.open_spectrum(spec_row.spectrum_id)
    except Exception as e:
        result['status'] = 'skip'
        result['message'] = f"Spectrum retrieval FAILED: {e}"
        return result

    wave_obs = spec.wavelength
    flux_raw = spec.flam        # erg/s/cm^2/A -- matches this module's flux-unit assumptions
    err_raw = spec.flam_err
    good = spec.valid            # canonical finite-flux & positive-error mask

    wave_c = wave_obs[good]
    flux_c = flux_raw[good]
    err_c = err_raw[good]

    if len(wave_c) < 10:
        result['status'] = 'skip'
        result['message'] = "Too few good pixels."
        return result

    if not (wave_c.min() < HBETA_obs < wave_c.max()):
        result['status'] = 'skip'
        result['message'] = "Hβ not in wavelength coverage."
        return result
    if not (wave_c.min() < HALPHA_obs < wave_c.max()):
        result['status'] = 'skip'
        result['message'] = "Hα not in wavelength coverage."
        return result

    seg = (wave_c > HBETA_obs - 0.08) & (wave_c < SII_6731 + 0.08)
    wave_seg = wave_c[seg]
    flux_seg = flux_c[seg]
    err_seg = err_c[seg]

    if len(wave_seg) < 10:
        result['status'] = 'skip'
        result['message'] = "Too few pixels in segment."
        return result

    try:
        wave_fit, flux_fit, err_fit, df, fit_flags = line_fitting(
            wave_seg, flux_seg, err_seg,
            R_interp=R_interp,
            hb_center=HBETA_obs, oiii_center=OIII_obs, ha_center=HALPHA_obs,
            window=window, window_n_sigma=window_n_sigma, window_max=window_max,
            diagnose=False,
            **line_fit_kwargs,
        )
    except ValueError as e:
        result['status'] = 'skip'
        result['message'] = f"Skipping (coverage/setup): {e}"
        return result
    except Exception as e:
        result['status'] = 'error'
        result['message'] = f"Fitting FAILED: {e}\n{traceback.format_exc()}"
        return result

    try:
        summ = posterior_summary(df)
        data = {
            'Hb_flux': summ['Flux_Hb'], 'Hb_err': summ['Flux_Hb_err'],
            'Ha_flux': summ['Flux_Ha'], 'Ha_err': summ['Flux_Ha_err'],
            'OIII_flux': summ['Flux_OIII'], 'OIII_err': summ['Flux_OIII_err'],
            'SII_flux': summ['Flux_SII'], 'SII_err': summ['Flux_SII_err'],
            'Hb_SNR': summ['Flux_Hb_snr'], 'Ha_SNR': summ['Flux_Ha_snr'],
            'fit_nii': int(fit_flags['fit_nii']), 'fit_sii': int(fit_flags['fit_sii']),
            'snr_nii_prefit': fit_flags['snr_nii'], 'snr_sii_prefit': fit_flags['snr_sii'],
        }

        if not fit_flags['fit_sii']:
            data['SII_flux'] = np.nan
            data['SII_err'] = np.nan

        data['A_ha_snr'] = np.median(df['A_ha']) / (
            (np.percentile(df['A_ha'], 84) - np.percentile(df['A_ha'], 16)) / 2
        )

        med_params = df.quantile(0.5)[
            ['A_hb', 'A_oiii', 'A_ha', 'mu_ha', 'R_nii', 'R_sii', 'sigma_int', 'm', 'b']
        ].values
        residuals = (flux_fit - full_line_model(
            wave_fit, *med_params, R_interp=R_interp,
            fit_nii=fit_flags['fit_nii'], fit_sii=fit_flags['fit_sii']
        )) / err_fit
        data['chi2_red'] = np.sum(residuals ** 2) / (len(flux_fit) - 9)

        data['Ha_flux_p16'] = summ['Flux_Ha_p16']
        data['Hb_flux_p16'] = summ['Flux_Hb_p16']
        data['Ha_flux_p84'] = summ['Flux_Ha_p84']
        data['Hb_flux_p84'] = summ['Flux_Hb_p84']

        bd = df['Flux_Ha'] / df['Flux_Hb'].replace(0, np.nan)
        bd_clean = bd.dropna()
        data['Balmer_dec'] = np.median(bd_clean)
        data['Balmer_dec_err'] = (np.percentile(bd_clean, 84) - np.percentile(bd_clean, 16)) / 2

        # E(B-V) using Calzetti+2000 attenuation law:
        #   E(B-V) = 1.97 * log10((Ha/Hb) / 2.86)   [2.86 = Case B Balmer decrement]
        with np.errstate(invalid='ignore', divide='ignore'):
            ebv = 1.97 * np.log10(bd / 2.86)
        ebv_clean = ebv.dropna()
        data['EBV'] = np.median(ebv_clean)
        data['EBV_err'] = (np.percentile(ebv_clean, 84) - np.percentile(ebv_clean, 16)) / 2

        data['effective_steps'] = fit_flags['effective_steps']
        data['acc_frac'] = fit_flags['acc_frac']
        data['converged'] = int(fit_flags.get('curve_fit_converged', True))
        data['reliable'] = int(fit_flags.get('reliable', False))

        result['data'] = data
    except Exception as e:
        result['status'] = 'error'
        result['message'] = f"Result extraction FAILED: {e}"
        return result

    if plot_dir is not None:
        try:
            med_params = df.quantile(0.5)[
                ['A_hb', 'A_oiii', 'A_ha', 'mu_ha', 'R_nii', 'R_sii', 'sigma_int', 'm', 'b']
            ].values
            xarr = np.linspace(HBETA_obs - 0.1, SII_6731 + 0.1, 1200)
            model_curve = full_line_model(
                xarr, *med_params, R_interp=R_interp,
                fit_nii=fit_flags['fit_nii'], fit_sii=fit_flags['fit_sii']
            )

            fig, ax = plt.subplots(figsize=(11, 5))
            ax.step(wave_fit, flux_fit, where='mid', color='black', alpha=0.6, label='Data')
            ax.fill_between(wave_fit, flux_fit - err_fit, flux_fit + err_fit, alpha=0.2, color='grey')
            ax.plot(xarr, model_curve, color='royalblue', lw=1.8, label='Model')

            for lam, lbl, col, ls in [
                (HBETA_obs, 'Hβ', 'steelblue', '--'),
                (OIII_4959, '[OIII]4959', 'olivedrab', ':'),
                (OIII_obs, '[OIII]5007', 'seagreen', '--'),
                (HALPHA_obs, 'Hα', 'mediumpurple', '--'),
                (SII_6716, '[SII]6716', 'darkorange', '--'),
                (SII_6731, '[SII]6731', 'darkorange', ':'),
                (NII_6548, '[NII]6548', 'crimson', '--'),
                (NII_6583, '[NII]6583', 'crimson', ':'),
            ]:
                ax.axvline(lam, color=col, ls=ls, lw=0.8, label=lbl)

            flag_str = (f"NII={'fit' if fit_flags['fit_nii'] else 'fixed=0'} "
                        f"(SNR={fit_flags['snr_nii']:.1f})  |  "
                        f"SII={'fit' if fit_flags['fit_sii'] else 'fixed=0'} "
                        f"(SNR={fit_flags['snr_sii']:.1f}, "
                        f"res={'Y' if fit_flags['sii_resolvable'] else 'N'})")
            ax.set_title(f"ID {spec_id}   z={z:.4f}\n{flag_str}", fontsize=9)
            ax.legend(fontsize=7, ncol=4)
            ax.set_xlabel('Wavelength (µm)')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"fit_{spec_id}.png"), dpi=120)
            plt.close(fig)
        except Exception as e:
            result['plot_message'] = f"Plot FAILED: {e}"

    return result


def run_photspec_batch(catalog_df, R_interp, output_csv, plot_dir=None,
                        n_processes=8, checkpoint_every=10,
                        window=0.03, window_n_sigma=8, window_max=0.05,
                        grating='PRISM', campfire_data_dir=None,
                        **line_fit_kwargs):
    """
    Fit emission lines for every row of a photspec catalog DataFrame
    (columns: object_id, z_spec, ID_spec), fitting up to n_processes
    galaxies concurrently -- one per core.

    Parameters
    ----------
    catalog_df : pd.DataFrame
        Must have 'object_id', 'z_spec', 'ID_spec' columns.
    R_interp : callable
        Spectral resolution interpolator from load_instrument_lsf().
    output_csv : str
        Path to write the results CSV (checkpointed periodically).
    plot_dir : str or None
        If given, diagnostic plots are saved here as fit_{ID_spec}.png.
    n_processes : int
        Number of worker processes.
    checkpoint_every : int
        Write output_csv to disk after this many completed rows.
    window, window_n_sigma, window_max :
        Passed through to the per-row fit (see line_fitting/line_window).
    grating : str
        Which grating's spectrum to fit for each object (default 'PRISM').
        If an object has more than one spectrum in this grating (e.g.
        multiple programs/visits), the highest-S/N one is used.
    campfire_data_dir : str or None
        Explicit Campfire data directory (contains meta/ and products/),
        passed to Campfire(data_dir=...) in every worker process.
    **line_fit_kwargs :
        Any other kwargs to pass through to line_fitting (e.g. steps,
        burnin, nwalkers, snr_thresh_nii, snr_thresh_sii).

    Returns
    -------
    df_out : pd.DataFrame
        catalog_df with all result columns filled in.
    """
    df_out = catalog_df.copy()
    result_cols = [
        'Hb_flux', 'Hb_err', 'Ha_flux', 'Ha_err', 'OIII_flux', 'OIII_err', 'SII_flux', 'SII_err',
        'Hb_SNR', 'Ha_SNR', 'fit_nii', 'fit_sii', 'snr_nii_prefit', 'snr_sii_prefit',
        'A_ha_snr', 'chi2_red', 'Ha_flux_p16', 'Hb_flux_p16', 'Ha_flux_p84', 'Hb_flux_p84',
        'Balmer_dec', 'Balmer_dec_err', 'EBV', 'EBV_err', 'converged', 'reliable', 'acc_frac',
        'effective_steps',
    ]
    for col in result_cols:
        if col not in df_out.columns:
            df_out[col] = np.nan

    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)

    worker = partial(
        _process_photspec_row, R_interp=R_interp, window=window,
        window_n_sigma=window_n_sigma, window_max=window_max,
        plot_dir=plot_dir, line_fit_kwargs=line_fit_kwargs, grating=grating,
    )

    n_total = len(df_out)
    n_done = 0
    with Pool(processes=n_processes, initializer=_init_campfire_worker,
              initargs=(campfire_data_dir,)) as pool:
        for result in pool.imap_unordered(worker, df_out.iterrows()):
            spec_id = result['spec_id']
            n_done += 1
            print(f"\n{'='*60}\n[{n_done}/{n_total}] ID {spec_id}: {result['status']}"
                  + (f" -- {result['message']}" if result['message'] else ""))

            if result['status'] == 'ok' and result['data'] is not None:
                for k, v in result['data'].items():
                    df_out.loc[result['idx'], k] = v
            if result.get('plot_message'):
                print(f"  → {result['plot_message']}")

            if n_done % checkpoint_every == 0:
                df_out.to_csv(output_csv, index=False)
                print(f"  → Checkpoint saved ({n_done}/{n_total})")

    df_out.to_csv(output_csv, index=False)
    print(f"\nDone. Results written to {output_csv}")
    return df_out