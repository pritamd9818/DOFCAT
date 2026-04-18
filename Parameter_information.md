# Tweakable Parameters

The DOFCAT pipeline includes several parameters that influence preprocessing, visualization, and velocity estimation. These parameters should be tuned depending on the dataset, noise level, and scientific objective.

---

### 1. Brightness Scaling (vmin / vmax) — Most Important

Defined in preprocessing scripts.

#### METIS

```python
vmin = np.min([np.nanmin(img) * np.exp(-11.0) for img in diff_imgs])
vmax = np.max([np.nanmax(img) * np.exp(-6.0) for img in diff_imgs])

Or, simply use:
    
    vmin = -1.936907808319677e-12
    vmax = 2.9202553655386404e-10

```

#### ASPIICS

```python
vmin = np.min(all_vals) * np.exp(-21.2)
vmax = np.max(all_vals) * np.exp(-17.8)

Or, simply use:
    
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
* Should be tuned per event for best visibility

---

### 2. Running Difference Interval

```python
diff = img[i+2] - img[i]
```

**Purpose:**
Defines temporal baseline for motion detection.

**Effect:**

* Larger interval → enhances large-scale motion, increases signal
* Smaller interval → captures finer temporal evolution

---

### 3. Bilateral Filtering (Noise Reduction)

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

### 4. ROI (Region of Interest)

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

### 5. Farneback Optical Flow Parameters

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

### 6. Velocity Thresholding

```python
lower_velocity = 50
upper_velocity = 1500
```

**Purpose:**
Removes noise and unphysical velocities.

**Effect:**

* Lower threshold → removes slow background motion
* Upper threshold → removes spurious high-speed artifacts

**Important:**

* Strongly influences final velocity maps
* Should be tuned based on CME speed

---

### 7. Velocity Conversion Parameters

Derived from FITS headers:

```python
cdelt = headers[0]['CDELT1']
dsun_obs = headers[0]['DSUN_OBS']
```

**Purpose:**
Convert pixel displacement → physical velocity (km/s)

---

### 8. Temporal GTF Filter (ASPIICS)

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

### 9. Masking Parameters

#### METIS

* Uses: `INN_FOV`, `OUT_FOV`, `SUNPIX`, `IOPIX`

#### ASPIICS

```python
rin_rsun = 1.6
```

**Purpose:**
Removes occulter and unwanted regions.

---

### 10. Quiver Plot Density

```python
step = 15
```

**Purpose:**
Controls arrow spacing.

* Smaller → dense arrows (cluttered)
* Larger → cleaner visualization

---

### 11. Arrow Scaling

```python
arrow_scale = 4
```

**Purpose:**
Controls vector length in overlay plots.

---

### 12. Plotting and Layout

```python
dpi = 100
padding = 200
```

**Purpose:**
Ensures consistent figure size and layout.

---

### 13. Output Control

```python
plot_heatmaps = True
plot_vectors = True
```

**Purpose:**
Enable/disable plotting.

---

## Notes

* Brightness scaling (`vmin`, `vmax`) is the most sensitive visualization parameter.
* Optical flow parameters control the physical accuracy of velocity estimation.
* Preprocessing (filtering + masking) strongly affects final results.
* Parameters may need tuning for different CME events.

---
