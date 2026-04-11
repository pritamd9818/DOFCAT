'''
Written by: Pritam Das, ARIES, Nainital, India
Date: 2026-03-31
Version: DOFCAT Optical Flow 1.0

Description: This script takes the running difference images generated from preprocessing of the data and
applies dense optical flow techniques to compute the velocity field of the observed CME. The main steps include:
1. Loading the preprocessed difference images.
2. Applying Bilateral filter to reduce noise while preserving CME features.
3. Extracting a region of interest (ROI). Change it as per the requirement. Larger ROIs are more computationally expensive, but captures more motion.
4. Computing dense optical flow using Farneback's method to get pixel displacements.
5. Converting pixel displacements into physical velocities (km/s) using the FITS header information.
6. Applying noise filtering to retain only physically meaningful velocities.
7. Plotting the velocity magnitude as a heatmap with overlaid quiver vectors to visualize the flow field.
8. Saving the resulting visualizations for further analysis.

'''



import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import pickle
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.time import Time


# ------------------------------------------------------------
# Load PNG frames (difference images)
# ------------------------------------------------------------
def load_frames_from_folder(folder_path):

    # Collect all PNG files and sort them chronologically
    file_paths = sorted(glob(os.path.join(folder_path, '*.png')))

    frames = []
    for file_path in file_paths:
        frame = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # Skip corrupted/unreadable files
        if frame is not None:
            frames.append(frame)

    return frames


# ------------------------------------------------------------
# Basic denoising to reduce small-scale noise
# ------------------------------------------------------------
def reduce_noise(img):

    # Bilateral filter preserves edges while smoothing
    denoised = cv2.bilateralFilter(img, 35, 60, 60)
    return denoised


# ------------------------------------------------------------
# Extract Region of Interest (ROI)
# ------------------------------------------------------------
def set_ROI(frame, x_value, y_value, width, height):

    # Simple rectangular crop
    return frame[y_value:y_value + height, x_value:x_value + width]


# ------------------------------------------------------------
# Extract observation time from FITS header
# ------------------------------------------------------------
def extract_datetime_from_header(header):

    try:
        date_obs = str(header.get('DATE-OBS', header.get('DATE', ''))).strip()

        # Handle missing or malformed entries
        if not date_obs or date_obs.startswith('.'):
            raise ValueError

        # Normalize format if needed
        if '/' in date_obs:
            date_obs = date_obs.replace('/', '-')

        date_time = Time(date_obs, format='isot', scale='utc')

        return date_time.strftime('%Y-%m-%d %H:%M:%S UT')

    except Exception:
        return "Unknown Time"


# ------------------------------------------------------------
# Optical flow computation (core algorithm untouched)
# ------------------------------------------------------------
def compute_optical_flow_and_magnitude(frames, headers,
                                       x_value, y_value, width, height,
                                       lower_velocity=50, upper_velocity=3000):

    magnitudes = []
    u_list = []
    v_list = []

    # First frame ROI
    prev_frame = set_ROI(frames[0], x_value, y_value, width, height)

    for i in range(1, len(frames)):

        # Next frame ROI
        next_frame = set_ROI(frames[i], x_value, y_value, width, height)

        # Dense optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None,
                                            0.3, 4, 15, 4, 5, 1.1, 0)

        u = flow[..., 0]
        v = flow[..., 1]

        # Pixel displacement magnitude
        velocity = np.sqrt(u**2 + v**2)

        # --------------------------------------------------------
        # Convert pixel displacement -> physical velocity (km/s)
        # --------------------------------------------------------

        cdelt = headers[0]['CDELT1']      # arcsec/pixel
        dsun_obs_m = headers[0]['DSUN_OBS']  # Sun-observer distance (m)

        arcsec_to_radian = np.pi / (180.0 * 3600.0)

        # Convert pixel scale into radians
        pixel_scale_radian = cdelt * arcsec_to_radian

        # Physical size per pixel (km)
        D = (pixel_scale_radian * dsun_obs_m) / 1e3

        # --------------------------------------------------------
        # Time cadence between frames
        # --------------------------------------------------------

        t1_str = extract_datetime_from_header(headers[i - 1])
        t2_str = extract_datetime_from_header(headers[i])

        t1 = Time(t1_str.replace(" UT", ""), format='iso')
        t2 = Time(t2_str.replace(" UT", ""), format='iso')

        F = (t2 - t1).sec

        print(f"Cadence time (F): {F} seconds")

        # Final velocity (km/s)
        magnitude = (velocity * D) / F

        # --------------------------------------------------------
        # Noise filtering (unchanged logic)
        # --------------------------------------------------------

        # Keep physically meaningful velocities
        velocity_mask = (magnitude > lower_velocity) & (magnitude < upper_velocity)

        u_filtered = np.where(velocity_mask, u, 0)
        v_filtered = np.where(velocity_mask, v, 0)
        magnitude_filtered = np.where(velocity_mask, magnitude, 0)

        magnitudes.append(magnitude_filtered)
        u_list.append(u_filtered)
        v_list.append(v_filtered)

        prev_frame = next_frame

    return magnitudes, u_list, v_list


# ------------------------------------------------------------
# Heatmap + quiver plot
# ------------------------------------------------------------
def plot_velocity_heatmap_with_quiver(magnitude, u, v, output_dir, headers,
                                      step=30, frame_number=0, date_time_str=""):

    plt.figure(figsize=(10.24, 10.24), dpi=300)

    # Modify turbo colormap so zero appears black
    turbo = plt.cm.turbo(np.linspace(0, 1, 256))
    turbo[0] = [0, 0, 0, 1]
    black_turbo = ListedColormap(turbo)

    plt.imshow(magnitude, cmap=black_turbo, interpolation='none')

    plt.colorbar(label='Velocity Magnitude (km/s)',
                 orientation='horizontal', fraction=0.04, aspect=18, pad=0.1)

    plt.title(f'Velocity Magnitude of CME\n{date_time_str}')

    # --------------------------------------------------------
    # Axis scaling in solar radii
    # --------------------------------------------------------

    cdelt = headers[0]['cdelt1']
    center_x = magnitude.shape[1] // 2
    center_y = magnitude.shape[0] // 2

    x_extent_arcsec = (np.array([0, magnitude.shape[1] - 1]) - center_x) * cdelt
    y_extent_arcsec = (np.array([0, magnitude.shape[0] - 1]) - center_y) * cdelt

    rsun_arcsec = headers[0]['RSUN_ARC']

    x_rs = x_extent_arcsec / rsun_arcsec
    y_rs = y_extent_arcsec / rsun_arcsec

    plt.xlabel('Solar X ($R_\\odot$)')
    plt.ylabel('Solar Y ($R_\\odot$)')

    # --------------------------------------------------------
    # Subsample arrows for clarity
    # --------------------------------------------------------

    y, x = np.mgrid[0:magnitude.shape[0]:step, 0:magnitude.shape[1]:step]

    u_sub = u[::step, ::step]
    v_sub = v[::step, ::step]

    plt.quiver(x, y, u_sub, -v_sub, color='white',
               scale=500, alpha=0.6, width=0.0025)

    # Save output
    os.makedirs(os.path.join(output_dir, "intensity"), exist_ok=True)

    plt.savefig(os.path.join(output_dir, "intensity",
                             f"frame_{frame_number:04d}.png"),
                dpi=100, bbox_inches='tight', pad_inches=0)

    plt.close()


# ------------------------------------------------------------
# Overlay vectors on original frames
# ------------------------------------------------------------
def plot_velocity_vectors_on_original_frames(frames, magnitudes,
                                             u_list, v_list,
                                             output_dir, step=60,
                                             x_value=0, y_value=0):

    os.makedirs(os.path.join(output_dir, "frames_with_vectors"), exist_ok=True)

    for i, (frame, magnitude, u, v) in enumerate(zip(frames, magnitudes, u_list, v_list)):

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        # Apply same ROI used in OF
        roi = set_ROI(frame_rgb, x_value, y_value,
                      magnitude.shape[1], magnitude.shape[0])

        y, x = np.mgrid[0:magnitude.shape[0]:step, 0:magnitude.shape[1]:step]

        norm = plt.Normalize(vmin=np.min(magnitude), vmax=np.max(magnitude))

        turbo = plt.cm.turbo(np.linspace(0, 1, 256))
        turbo[0] = [0, 0, 0, 1]
        colormap = ListedColormap(turbo)

        arrow_scale = 4

        for j in range(x.shape[0]):
            for k in range(x.shape[1]):

                y_pos = y[j, k]
                x_pos = x[j, k]

                dx = int(u[y_pos, x_pos] * arrow_scale)
                dy = int(v[y_pos, x_pos] * arrow_scale)

                color = tuple(int(255 * c) for c in colormap(norm(magnitude[y_pos, x_pos]))[:3])

                cv2.arrowedLine(roi,
                                (x_pos, y_pos),
                                (x_pos + dx, y_pos + dy),
                                color, 2, tipLength=0.4)

        plt.figure(figsize=(10.24, 10.24), dpi=200)
        plt.imshow(frame_rgb)
        plt.axis('off')

        # Add colorbar
        sm = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
        cbar = plt.colorbar(sm, ax=plt.gca(), orientation= 'horizontal', fraction=0.04,aspect=18, pad=0.00001)
        cbar.set_label('Velocity Magnitude (km/s)', fontsize=10)

        ticks = np.linspace(np.min(magnitude), np.max(magnitude), num=5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{tick:.2f}" for tick in ticks])
        cbar.ax.tick_params(labelsize=8)
        plt.subplots_adjust(right=1.2)  # <--- Add this line

        plt.savefig(os.path.join(output_dir, "frames_with_vectors",
                                 f"frame_{i:04d}.png"),
                    bbox_inches='tight', pad_inches=0)

        plt.close()


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    frames_folder = "/full/path/to/difference_images"
    output_dir = "/full/path/to/output"

    noisy_frames = load_frames_from_folder(frames_folder)

    # Apply denoising to each frame
    denoised_frames = [reduce_noise(frame) for frame in noisy_frames]

    with open(f"{frames_folder}/difference_headers.pkl", "rb") as f:
        headers = pickle.load(f)

    padding = 200
    pad = padding // 2   # = 100 pixels each side

    full_height, full_width = denoised_frames[0].shape

    # Start ROI after padding
    x_value = pad
    y_value = pad

    # Extract only actual data region
    width = full_width - 2 * pad
    height = full_height - 2 * pad

    # --------------------------------------------------------

    magnitudes, u_list, v_list = compute_optical_flow_and_magnitude(
        denoised_frames, headers,
        x_value, y_value, width, height,
        lower_velocity=50, upper_velocity=3000
    )

    # Plot heatmaps
    for i, (magnitude, u, v) in enumerate(zip(magnitudes, u_list, v_list)):
        date_time_str = extract_datetime_from_header(headers[i])

        plot_velocity_heatmap_with_quiver(
            magnitude, u, v, output_dir, headers,
            step=15, frame_number=i, date_time_str=date_time_str
        )

    # Overlay vectors
    plot_velocity_vectors_on_original_frames(
        noisy_frames, magnitudes, u_list, v_list,
        output_dir, step=15,
        x_value=x_value, y_value=y_value
    )

    return magnitudes


if __name__ == "__main__":
    magnitudes = main()
