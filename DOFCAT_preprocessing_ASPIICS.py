'''
Written by: Pritam Das, ARIES, Nainital, India
Date: 2026-03-31
Version: DOFCAT Preprocessing 2.0 [ASPIICS]

Description: This script processes PROBA-3/ASPIICS Wideband 10 second exposure Level 2 FITS files to create running difference images. It performs the following steps:
1. Loads all FITS files from a specified folder.
2. Creates running difference images by subtracting frame i from frame i+2.
3. Applies a temporal Gaussian Tapered Fourier (GTF) filter to mitigate brightness flickering.
4. Rotates each difference image to solar north up using SunPy's Map rotation.
5. Applies an annular mask to focus on the region of interest (1.6 Rsun to the outer edge).
6. Plots and saves the final processed difference images with solar disk overlay and consistent scaling.
7. Saves the difference images in a compressed format for future use.

'''

import os
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.time import Time
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import colormaps
import sunpy.visualization.colormaps as cm
from sunpy.map import Map
import pickle


# ------------------------------------------------------------
# Load FITS files
# ------------------------------------------------------------
def open_proba_fits_files(path):
    proba_all = []
    header_all = []

    # Collect all files and filter FITS only
    files = sorted([f for f in Path(path).glob("*") if f.is_file()])
    valid_files = [f for f in files if f.suffix.lower() in ['.fts', '.fits']]

    if len(valid_files) == 0:
        raise FileNotFoundError(f"No .fts or .fits files found in {path}")

    # Read data + header from each file
    for location in valid_files:
        with fits.open(location) as a:
            proba_all.append(a[0].data.astype(np.float32))
            header_all.append(a[0].header)

    return proba_all, header_all


# ------------------------------------------------------------
# Mask (annulus in Rsun)
# ------------------------------------------------------------
def create_proba_mask(header, shape, rin_rsun=1.3, rout_rsun=None):

    n_rows, n_cols = shape

    # WCS parameters needed to locate Sun center
    cdelt1 = float(header['CDELT1'])
    cdelt2 = float(header['CDELT2'])
    crpix1 = float(header['CRPIX1'])
    crpix2 = float(header['CRPIX2'])
    crval1 = float(header['CRVAL1'])
    crval2 = float(header['CRVAL2'])
    rsun_arc = float(header['RSUN_ARC'])

    # Convert Sun center to pixel coordinates
    sun_xcen = crpix1 - (crval1 / cdelt1)
    sun_ycen = crpix2 - (crval2 / cdelt2)

    # Convert Rsun into pixel scale
    rsun_pix = rsun_arc / ((cdelt1 + cdelt2) / 2.0)

    # Inner mask radius (removes occulter region)
    rin_pix = rin_rsun * rsun_pix

    # Outer radius: full image unless specified
    if rout_rsun is None:
        corners = np.array([[0,0],[n_cols-1,0],[0,n_rows-1],[n_cols-1,n_rows-1]])

        # Max distance from Sun center to corners
        rout_pix = np.max(np.hypot(corners[:,0]-sun_xcen,
                                  corners[:,1]-sun_ycen))
    else:
        rout_pix = rout_rsun * rsun_pix

    # Build distance map
    yy, xx = np.indices((n_rows, n_cols))
    dist = np.sqrt((xx - sun_xcen)**2 + (yy - sun_ycen)**2)

    # Keep only annular region
    return (dist >= rin_pix) & (dist <= rout_pix)


# ------------------------------------------------------------
# Coordinate extent
# ------------------------------------------------------------
def get_extent_info(header, shape):

    # Standard WCS conversion from pixel to world coordinates
    crval1 = float(header['CRVAL1'])
    crval2 = float(header['CRVAL2'])
    cdelt1 = float(header['CDELT1'])
    cdelt2 = float(header['CDELT2'])
    crpix1 = float(header['CRPIX1'])
    crpix2 = float(header['CRPIX2'])

    # Lower-left corner
    orgn = [crval1 - cdelt1 * crpix1,
            crval2 - cdelt2 * crpix2]

    # Upper-right corner
    ep = [orgn[0] + cdelt1 * shape[1],
          orgn[1] + cdelt2 * shape[0]]

    # Needed to avoid distortion in plotting
    aspect_ratio = (ep[0] - orgn[0]) / (ep[1] - orgn[1])

    return orgn, ep, aspect_ratio


# ------------------------------------------------------------
# Solar disk overlay
# ------------------------------------------------------------
def add_solar_disk_circle(ax, header):

    # Extract WCS parameters again
    crval1 = float(header['CRVAL1'])
    crval2 = float(header['CRVAL2'])
    cdelt1 = float(header['CDELT1'])
    cdelt2 = float(header['CDELT2'])
    crpix1 = float(header['CRPIX1'])
    crpix2 = float(header['CRPIX2'])
    rsun_arcsec = float(header['RSUN_ARC'])

    # Sun center in pixel coordinates
    sun_xcen = crpix1 - (crval1 / cdelt1)
    sun_ycen = crpix2 - (crval2 / cdelt2)

    # Convert to world coordinates
    x_world = crval1 + (sun_xcen - crpix1) * cdelt1
    y_world = crval2 + (sun_ycen - crpix2) * cdelt2

    # Convert to solar radii units
    x_rs = x_world / rsun_arcsec
    y_rs = y_world / rsun_arcsec

    # Draw circle of radius = 1 Rsun
    circle = Circle((x_rs, y_rs), radius=1.0,
                    edgecolor='white', facecolor='none',
                    linewidth=1.5, linestyle='--')

    ax.add_patch(circle)


# ------------------------------------------------------------
# Temporal GTF Filter
# ------------------------------------------------------------
def temporal_fft_filter(diff_cube, f_cut_ratio, sigma_ratio,
                        chunk_size, window_type, max_fft_bytes_safe):

    from scipy.signal import get_window

    # Dimensions: time × y × x
    N_diff, Ny, Nx = diff_cube.shape

    # Window applied before FFT to reduce edge effects
    window = get_window(window_type, N_diff).astype(np.float32)

    # Frequency axis (temporal frequencies)
    freqs = np.fft.rfftfreq(N_diff, d=1.0)
    nyquist = 0.5

    # Width of Gaussian filter
    if sigma_ratio is None:
        sigma_ratio = f_cut_ratio

    f_sigma = sigma_ratio * nyquist

    # Gaussian weighting in frequency domain
    gauss = np.exp(-0.5 * (freqs / (f_sigma + 1e-20))**2).astype(np.float32)
    gauss_mask = gauss[:, None, None]

    filtered_cube = np.empty_like(diff_cube, dtype=np.float32)

    eps = 1e-8

    # Try to use scipy FFT (faster)
    try:
        from scipy import fft as sfft
        have_scipy_fft = True
    except Exception:
        sfft = None
        have_scipy_fft = False

    # Estimate memory usage for FFT
    def _estimate_block_fft_bytes(n_time, chunk_h, nx):
        n_freq = n_time // 2 + 1
        return n_freq * chunk_h * nx * 16

    # Process in chunks along y (avoids large memory spikes)
    for y0 in range(0, Ny, chunk_size):
        y1 = min(y0 + chunk_size, Ny)

        block = diff_cube[:, y0:y1, :].astype(np.float32)

        # Replace NaNs before FFT (FFT cannot handle NaNs)
        nanmask = ~np.isfinite(block)
        if np.any(nanmask):
            mean_block = np.nanmean(block, axis=0, keepdims=True)
            mean_block = np.where(np.isfinite(mean_block), mean_block, 0.0)
            block[nanmask] = np.broadcast_to(mean_block, block.shape)[nanmask]

        # Decide whether to use fast or safe FFT
        est_bytes = _estimate_block_fft_bytes(N_diff, y1-y0, Nx)
        use_fast = est_bytes <= max_fft_bytes_safe and have_scipy_fft

        if use_fast:
            # Apply window
            block_w = block * window[:, None, None]

            # FFT → filter → inverse FFT
            fft_vals = sfft.rfft(block_w, axis=0, workers=1)
            fft_vals *= gauss_mask
            block_ifft = sfft.irfft(fft_vals, n=N_diff, axis=0, workers=1)

            # Remove window effect
            denom = np.where(np.abs(window[:, None, None]) > eps,
                             window[:, None, None], 1.0)
            block_filtered = block_ifft / denom

        else:
            # Fallback: column-wise FFT (slower but safer)
            block_filtered = np.empty_like(block, dtype=np.float32)
            denom_safe = np.where(np.abs(window) > eps, window, 1.0)

            for x in range(Nx):
                col = block[:, :, x]

                col_w = col * window[:, None]
                fft_col = np.fft.rfft(col_w, axis=0)
                fft_col *= gauss[:, None]

                col_ifft = np.fft.irfft(fft_col, n=N_diff, axis=0)
                block_filtered[:, :, x] = col_ifft / denom_safe[:, None]

        # Restore original NaN locations
        orig_nan = ~np.isfinite(diff_cube[:, y0:y1, :])
        block_filtered[orig_nan] = np.nan

        filtered_cube[:, y0:y1, :] = block_filtered

    return filtered_cube


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
def plot_all_proba_diff_images(diff_imgs, header_all, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "difference_images"), exist_ok=True)

    cmap = colormaps.get_cmap('kcor').copy()
    cmap.set_bad(color='black')

    # Use image size from header instead of hardcoding
    ref_header = header_all[0]
    naxis1 = int(ref_header['NAXIS1'])
    naxis2 = int(ref_header['NAXIS2'])

    dpi = 100
    padding = 200
    image_size = max(naxis1, naxis2)

    figsize = ((image_size + padding)/dpi, (image_size + padding)/dpi)

    # Global scaling across all frames (keeps contrast consistent)
    all_vals = np.concatenate([
        img[np.isfinite(img)]
        for img in diff_imgs if np.isfinite(img).any()
    ])

    vmin = np.min(all_vals) * np.exp(-21.2)
    vmax = np.max(all_vals) * np.exp(-17.8)

    for i, (img, header) in enumerate(zip(diff_imgs, header_all)):

        # Extract timestamp
        t = Time(str(header['DATE-OBS']).strip(), format='isot')
        time_str = t.strftime('%Y-%m-%d %H:%M:%S UT')

        # Convert to solar radii coordinates
        orgn, ep, _ = get_extent_info(header, img.shape)
        rsun_arc = float(header['RSUN_ARC'])

        extent = [orgn[0]/rsun_arc, ep[0]/rsun_arc,
                  orgn[1]/rsun_arc, ep[1]/rsun_arc]

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        # Plot image
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
                  origin='lower', extent=extent)

        # Overlay solar disk
        add_solar_disk_circle(ax, header)

        ax.set_title(f'PROBA3 WB Frame {i:04d}\n{time_str}')

        plt.savefig(f"{save_dir}/difference_images/difference_image_{i:04d}.png")
        plt.close()

    # Save headers for later use
    with open(f"{save_dir}/difference_images/difference_headers.pkl", "wb") as f:
        pickle.dump(header_all, f)


# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------
def create_proba_running_difference(path, save_dir):

    proba_all, header_all = open_proba_fits_files(path)

    diff_imgs = []

    # Running difference using frame i and i+2
    for i in range(len(proba_all) - 2):

        img1 = proba_all[i]
        img2 = proba_all[i+2]

        # Only subtract valid pixels
        valid = np.isfinite(img1) & np.isfinite(img2)

        diff = np.full(img1.shape, np.nan, dtype=np.float32)
        diff[valid] = img2[valid] - img1[valid]

        # Keep only frames with meaningful signal
        if np.isfinite(diff).any():
            diff_imgs.append(diff)

    if not diff_imgs:
        print("No valid difference images generated.")
        return [], [], [], None, None

    # Stack into time cube (needed for FFT filtering)
    diff_cube = np.stack(diff_imgs, axis=0)

    # Apply temporal filtering
    filtered_cube = temporal_fft_filter(
        diff_cube,
        f_cut_ratio=0.15,
        sigma_ratio=None,
        chunk_size=128,
        window_type="hann",
        max_fft_bytes_safe=2 * 1024**3
    )

    # Convert back to list of frames
    diff_imgs_filtered = [filtered_cube[i] for i in range(filtered_cube.shape[0])]

    rotated_diffs = []
    rotated_headers = []

    # Rotate each frame to solar north up
    for k, diff in enumerate(diff_imgs_filtered):

        hdr = header_all[k+2]

        # SunPy rotation (handles WCS properly)
        m = Map(diff, hdr)
        m_rot = m.rotate(recenter=True, missing=np.nan)

        img_rot = m_rot.data.astype(np.float32)
        hdr_rot = m_rot.meta

        # Apply annular mask after rotation
        mask = create_proba_mask(hdr_rot, img_rot.shape, rin_rsun=1.6)
        img_rot[~mask] = np.nan

        rotated_diffs.append(img_rot)
        rotated_headers.append(hdr_rot)

    # Plot final images
    plot_all_proba_diff_images(rotated_diffs, rotated_headers, save_dir)

    # Return coordinate info for downstream use
    orgn, ep, _ = get_extent_info(rotated_headers[0], rotated_diffs[0].shape)

    return rotated_diffs, proba_all, rotated_headers, orgn, ep


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------
if __name__ == "__main__":

    path = "/full/path/to/input/data/"
    save_dir = "/full/path/to/output/results/"

    diff_imgs, proba_all, header_all, orgn, ep = create_proba_running_difference(path, save_dir)

    # Save filtered difference images
    if diff_imgs:
        np.savez_compressed(
            os.path.join(save_dir, "difference_images/diff_imgs.npz"),
            **{f"diff_{i:04d}": img.astype(np.float32)
               for i, img in enumerate(diff_imgs)}
        )
