'''
Written by: Pritam Das, ARIES, Nainital, India
Date: 2026-03-31
Version: DOFCAT Preprocessing 2.0 [METIS]

Description: This script processes METIS Total Brightness (TB) FITS files to create running difference images. It performs the following steps:
1. Loads all FITS files from a specified folder.
2. Builds a mask to isolate the useful field-of-view region.
3. Converts pixel coordinates into physical coordinates for accurate plotting.
4. Draws a circle representing the solar disk (1 Rsun) on the images.
5. Plots and saves all running difference images with consistent scaling and a KCor-style colormap.
6. Saves the difference images in a compressed format for future use.
'''

import os
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.time import Time
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import colormaps
import pickle


# Load all FITS files from a folder
def open_metis_fits_files(path):
    metis_all = []
    header_all = []

    # Collect all FITS files
    files = sorted(Path(path).glob("*"))
    valid_files = [f for f in files if f.suffix.lower() in ['.fts', '.fits']]

    if len(valid_files) == 0:
        raise FileNotFoundError(f"No .fts or .fits files found in {path}")

    # Read image data and headers
    for location in valid_files:
        with fits.open(location) as a:
            metis_all.append(a[0].data.astype(np.float32))
            header_all.append(a[0].header)

    return metis_all, header_all


# Build a mask to keep only the useful field-of-view region
def create_metis_mask(header, shape):
    n_rows, n_cols = shape

    # Pixel locations of Sun center and inner occulter
    sun_xcen = float(header['SUNPIX1'])
    sun_ycen = float(header['SUNPIX2'])
    io_xcen = float(header['IOPIX1'])
    io_ycen = float(header['IOPIX2'])

    # Convert FOV limits from arcsec to pixels
    inn_fov_arcsec = float(header['INN_FOV']) * 3600
    out_fov_arcsec = float(header['OUT_FOV']) * 3600

    cdelt1 = float(header['CDELT1'])
    cdelt2 = float(header['CDELT2'])

    inner_radius = inn_fov_arcsec / cdelt1
    outer_radius = out_fov_arcsec / cdelt2

    # Coordinate grid
    yy, xx = np.indices((n_rows, n_cols))

    # Distance maps
    dist_sun = np.sqrt((xx - sun_xcen)**2 + (yy - sun_ycen)**2)
    dist_io = np.sqrt((xx - io_xcen)**2 + (yy - io_ycen)**2)

    # Keep pixels outside occulter and inside outer boundary
    mask = (dist_io >= inner_radius) & (dist_sun <= outer_radius)

    return mask


# Convert pixel grid into physical coordinates for plotting
def get_extent_info(header, shape):
    crval1 = header['CRVAL1']
    crval2 = header['CRVAL2']
    cdelt1 = header['CDELT1']
    cdelt2 = header['CDELT2']
    crpix1 = header['CRPIX1']
    crpix2 = header['CRPIX2']

    # Lower-left corner in world coordinates
    orgn = [crval1 - cdelt1 * crpix1, crval2 - cdelt2 * crpix2]

    # Upper-right corner
    ep = [orgn[0] + cdelt1 * shape[1], orgn[1] + cdelt2 * shape[0]]

    # Needed so image is not stretched
    aspect_ratio = (ep[0] - orgn[0]) / (ep[1] - orgn[1])

    return orgn, ep, aspect_ratio


# Draw a circle representing the solar disk (1 Rsun)
def add_solar_disk_circle(ax, header):
    crval1 = header['CRVAL1']
    crval2 = header['CRVAL2']
    cdelt1 = header['CDELT1']
    cdelt2 = header['CDELT2']
    crpix1 = header['CRPIX1']
    crpix2 = header['CRPIX2']
    sun_xcen = header['SUNPIX1']
    sun_ycen = header['SUNPIX2']
    rsun_arcsec = float(header['RSUN_ARC'])

    # Convert Sun center into world coordinates
    x_world = crval1 + (sun_xcen - crpix1) * cdelt1
    y_world = crval2 + (sun_ycen - crpix2) * cdelt2

    # Express in solar radii
    x_rs = x_world / rsun_arcsec
    y_rs = y_world / rsun_arcsec

    circle = Circle(
        (x_rs, y_rs),
        radius=1.0,
        transform=ax.transData,
        edgecolor='white',
        facecolor='none',
        linewidth=1.5,
        linestyle='--'
    )

    ax.add_patch(circle)


# Plot and save all running difference images
def plot_all_metis_diff_images(diff_imgs, header_all, orgn, ep, aspect_ratio, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "difference_images"), exist_ok=True)

    # KCor-style colormap
    cmap_metis_tb = colormaps.get_cmap('kcor').copy()
    cmap_metis_tb.set_bad(color='black')

    # Use global scaling across frames
    vmin = np.min([np.nanmin(img) * np.exp(-11.0) for img in diff_imgs])
    vmax = np.max([np.nanmax(img) * np.exp(-6.0) for img in diff_imgs])
    '''
    Or, simply use:
    
    vmin = -1.936907808319677e-12
    vmax = 2.9202553655386404e-10

    This value provided good results. But it can be tailored according to the use case.
    '''

    # Get image size from header
    ref_header = header_all[0]
    naxis1 = int(ref_header['NAXIS1'])
    naxis2 = int(ref_header['NAXIS2'])

    dpi = 100
    padding = 200

    # Keep square layout for consistent output
    image_size = max(naxis1, naxis2)
    total_size = image_size + padding
    figsize = (total_size / dpi, total_size / dpi)

    for i, (diff_img, header) in enumerate(zip(diff_imgs, header_all[2:])):

        # Read observation time if available
        try:
            date_obs = str(header['DATE-OBS']).strip()
            t = Time(date_obs, format='isot', scale='utc')
            time_str = t.strftime('%Y-%m-%d %H:%M:%S UT')
        except Exception:
            time_str = "Unknown Time"

        fig = plt.figure(figsize=figsize, dpi=dpi)

        left = padding / 2 / total_size
        bottom = padding / 2 / total_size
        size = image_size / total_size

        ax = fig.add_axes((left, bottom, size, size))

        # Convert extent into solar radii
        rsun_arcsec = float(header['RSUN_ARC'])
        extent_rs = [val / rsun_arcsec for val in [orgn[0], ep[0], orgn[1], ep[1]]]

        ax.imshow(
            diff_img,
            cmap=cmap_metis_tb,
            vmin=vmin,
            vmax=vmax,
            origin='lower',
            extent=extent_rs,
            aspect=aspect_ratio
        )

        # Overlay solar disk
        add_solar_disk_circle(ax, header)

        ax.set_title(f'METIS TB Frame {i:04d}\n{time_str}', fontsize=24, pad=20)
        ax.set_xlabel(r'Solar X [R$_\odot$]', fontsize=22)
        ax.set_ylabel(r'Solar Y [R$_\odot$]', fontsize=22)
        ax.tick_params(labelsize=20)

        plt.savefig(
            f"{save_dir}/difference_images/difference_image_{i:04d}.png",
            dpi=dpi,
            pad_inches=0
        )
        plt.close()

    # Save headers for later use
    with open(f"{save_dir}/difference_images/difference_headers.pkl", "wb") as f:
        pickle.dump(header_all[2:], f)


# Main pipeline
def create_metis_running_difference(path, save_dir):

    metis_all, header_all = open_metis_fits_files(path)

    # Check exposure times
    exposure_times = [header.get('EXPTIME') for header in header_all]

    if len(set(exposure_times)) > 1:
        print("Warning: Exposure times vary between frames:")
        for i, exptime in enumerate(exposure_times):
            print(f"Frame {i:04d}: EXPTIME = {exptime}")
    else:
        print(f"All frames have same exposure time: {exposure_times[0]}")

    diff_imgs = []

    # Running difference using frame i and i+2
    for i in range(len(metis_all) - 2):
        img1 = metis_all[i]
        img2 = metis_all[i + 2]

        mask = create_metis_mask(header_all[i], img1.shape)

        diff = img2 - img1
        diff[~mask] = np.nan

        diff_imgs.append(diff)

    if not diff_imgs:
        print("No valid difference images generated.")
        return [], [], [], None, None

    orgn, ep, aspect_ratio = get_extent_info(header_all[0], diff_imgs[0].shape)

    plot_all_metis_diff_images(
        diff_imgs,
        header_all,
        orgn,
        ep,
        aspect_ratio,
        save_dir
    )

    return diff_imgs, metis_all, header_all, orgn, ep


if __name__ == "__main__":

    # Folder containing input FITS files
    path = "/full/path/to/input/data/"

    # Folder where results will be saved
    save_dir = "/full/path/to/output/results/"

    diff_imgs, metis_all, header_all, orgn, ep = create_metis_running_difference(path, save_dir)

    # Save all difference images in compressed format
    np.savez_compressed(
        os.path.join(save_dir, "difference_images/diff_imgs.npz"),
        **{f"diff_{i:04d}": img.astype(np.float32)
           for i, img in enumerate(diff_imgs)}
    )
