# Tweakable Parameters

The DOFCAT pipeline includes several parameters that influence preprocessing, visualization, and velocity estimation. These parameters should be tuned depending on the dataset, noise level, and scientific objective.

---

### 1. Brightness Scaling (vmin / vmax) — Preprocessing

Defined in preprocessing scripts.

#### METIS

```python
vmin = np.min([np.nanmin(img) * np.exp(-11.0) for img in diff_imgs])
vmax = np.max([np.nanmax(img) * np.exp(-6.0) for img in diff_imgs])
```
Or, put:

```text  
vmin = -1.936907808319677e-12
vmax = 2.9202553655386404e-10
```

#### ASPIICS

```python
vmin = np.min(all_vals) * np.exp(-21.2)
vmax = np.max(all_vals) * np.exp(-17.8)

```
Or, put:
```text
vmin = -6.397e-13
vmax = 1.074e-10
```

**Purpose:**
Controls contrast of running-difference images.

**Effect of tuning:**

* More negative exponent → darker background, enhances faint CME structures
* Less negative exponent → brighter image, may saturate bright regions

**Important:**

* This directly affects **visual interpretation of CME morphology**
* Should be tuned per event for best visibility and to improve the contrast of the desired feature.

---

### 2. Running Difference Interval - Preprocessing

```python
diff = img[i+2] - img[i]
```

**Purpose:**
Running difference is useful for enhancing moving features. For more details, refer to our paper.

**Effect:**

* Larger interval → enhances large-scale motion, but may introduce motion blur and feature broadening.
* Smaller interval → captures finer temporal evolution but may introduce dark artifact at the trailing edge of a moving feature.

---

### 3. Bilateral Filtering (Noise Reduction) - Optical Flow

Defined in the optical flow script.
```python
cv2.bilateralFilter(img, 35, 60, 60)
```

**Parameters:**

* `d = 35` → neighborhood diameter
* `sigmaColor = 60` → intensity similarity
* `sigmaSpace = 60` → spatial smoothing

**Purpose:**
Removes small-scale noise while preserving CME edges.

**Effect of tuning:**

* Larger values → stronger smoothing, may blur fine structures
* Smaller values → retains details but leaves noise

**Important:**

* Strongly affects optical flow stability
* Over-smoothing reduces velocity gradients

---

### 4. ROI (Region of Interest) - Optical Flow

```python
padding = 200
pad = padding // 2

x_value = pad
y_value = pad

width = full_width - 2 * pad
height = full_height - 2 * pad
```

**Purpose:**
Removes padded regions (labels, ticks, margins) from analysis.

**Effect:**

* Larger ROI → captures more CME structure, increases computation
* Smaller ROI → faster but may miss important features

---

### 5. Farneback Optical Flow Parameters - Optical Flow

```python
cv2.calcOpticalFlowFarneback(
    prev_frame, next_frame, None,
    0.3, 4, 15, 4, 5, 1.1, 0
)
```

**Parameters:**

* `pyr_scale = 0.3` → pyramid scaling (smaller = finer motion sensitivity)
* `levels = 4` → number of pyramid levels (higher = better large-scale capture)
* `winsize = 15` → averaging window (larger = smoother, smaller = detailed)
* `iterations = 4` → iterations per level (higher = more accurate, slower)
* `poly_n = 5` → pixel neighborhood size
* `poly_sigma = 1.1` → smoothing for polynomial expansion

**Purpose:**
Controls how motion is estimated between frames.

**Important:**

* Directly affects velocity accuracy
* Should be modified cautiously

---

### 6. Velocity Thresholding - Optical Flow

```python
lower_velocity = 50
upper_velocity = 1500
```

**Purpose:**
Removes noise and unphysical velocities.

**Effect:**

* Lower threshold → removes slow background motion
* Upper threshold → removes sudden instrument artifact motion.

**Important:**

* Strongly influences final velocity maps
* Should be tuned based on feature speed

---


### 7. Temporal GTF Filter (ASPIICS) - ASPIICS Preprocessing

```python
temporal_fft_filter(
    diff_cube,
    f_cut_ratio=0.15,
    sigma_ratio=None,
    chunk_size=128,
    window_type="hann",
    max_fft_bytes_safe=2 * 1024**3
)
```

**Parameters:**

* `f_cut_ratio = 0.15` → frequency cutoff (lower = stronger smoothing)
* `sigma_ratio` → Gaussian width in frequency space
* `window_type = "hann"` → reduces FFT edge artifacts
* `chunk_size = 128` → memory control
* `max_fft_bytes_safe` → switches FFT mode

**Purpose:**
Removes brightness flickering in ASPIICS data.

---


### 8. Arrow Scaling

```python
arrow_scale = 4
```

**Purpose:**
Controls vector length in overlay plots. For a good cadence dataset, the motion between two frames can be very minute, because of which the vector arrows might be too small to be visually resolvable. Hence, increasing the arrow scale multiplies the vector arrow length for visual clarity.

---


### 9. Output Control

```python
plot_heatmaps = True
plot_vectors = True
```

**Purpose:**
Enable/disable plotting. Disabling plotting makes the code much faster and more computationally efficient. However, plotting is recommended to confirm whether the code is detecting the desired motion.

---

## Notes

* Brightness scaling (`vmin`, `vmax`) is the most crucial visualization parameter.
* Optical flow parameters control the physical accuracy of velocity estimation.
* Preprocessing (filtering + masking) strongly affects final results.
* Parameters may need tuning for different use cases.

---

# While DOFCAT was designed for CME kinematics using coronagraphs, it can be modified to analyse any large-scale motions in the domain of solar physics.
