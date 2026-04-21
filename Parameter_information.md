# Tweakable Parameters

The DOFCAT pipeline includes several parameters that influence preprocessing, visualization, and velocity estimation. These parameters should be tuned depending on the dataset, noise level, and scientific objective.

---

### 1. Brightness Scaling (`vmin`, `vmax`) — Preprocessing (Critical)

Defined in preprocessing scripts for visualization of running-difference images.

#### METIS

```python
vmin = np.min([np.nanmin(img) * np.exp(-11.0) for img in diff_imgs])
vmax = np.max([np.nanmax(img) * np.exp(-6.0) for img in diff_imgs])
```

Or fixed values may be used:

```text
vmin = -1.936907808319677e-12
vmax = 2.9202553655386404e-10
```


#### ASPIICS

```python
vmin = np.min(all_vals) * np.exp(-21.2)
vmax = np.max(all_vals) * np.exp(-17.8)
```

Or:

```text
vmin = -6.397e-13
vmax = 1.074e-10
```


#### **Purpose**

These parameters define the intensity range mapped to the colormap:

* `vmin` → lower bound (dark end)
* `vmax` → upper bound (bright end)

All pixel values are scaled within `[vmin, vmax]`.


#### **Effect of tuning**

##### Changing `vmin` (lower limit)

* Decreasing `vmin` (more negative):

  * Expands dynamic range toward faint structures.
  * Enhances weak CME features (e.g., diffuse fronts, faint flows).
  * May suppress contrast in brighter regions.

* Increasing `vmin` (less negative):

  * Clips faint background variations.
  * Improves visibility of strong structures.
  * May completely hide weak features.

##### Changing `vmax` (upper limit)

* Decreasing `vmax`:

  * Prevents bright regions (e.g., CME core) from saturating.
  * Enhances contrast within bright structures.
  * May suppress very high-intensity features.

* Increasing `vmax`:

  * Preserves full intensity range.
  * But compresses contrast → features appear washed out.

#### **Best Practice**

* First adjust `vmin` → reveal faint CME structures.
* Then adjust `vmax` → prevent saturation of bright regions.


#### **Important Notes**

* This directly affects **visual interpretation of CME morphology**.
* Should be tuned per event for best visibility and to improve the contrast of the desired feature.
* Fixed values improve consistency across frames; dynamic scaling adapts to individual events.


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

This step is essential because optical flow is highly sensitive to noise and small intensity fluctuations.

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

* Strongly affects optical flow stability.
* Over-smoothing reduces velocity gradients.

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

* Larger ROI → captures more CME structure, increases computation.
* Smaller ROI → faster but may miss important features.

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

* Lower threshold → removes detection of noise-induced motion
* Upper threshold → removes detection of sudden instrument artifact-induced motion.

**Important:**

* Strongly influences final velocity maps
* Should be tuned based on feature speed

---


### 7. Temporal GTF Filter (ASPIICS) — Preprocessing

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

#### **Purpose**

Removes temporal brightness flickering in ASPIICS running-difference images by applying a **Gaussian-tapered filter in the frequency domain**.

Each pixel is treated as a time series and filtered independently.


#### **Parameter Details**

##### `f_cut_ratio = 0.15`

* Defines cutoff frequency as a fraction of the Nyquist frequency.
* Controls how much high-frequency variation is removed.

**Effect:**

* Lower value → stronger smoothing, removes flickering but may suppress real rapid evolution.
* Higher value → preserves temporal detail but retains noise.

##### `sigma_ratio`

* Controls the width of the Gaussian taper in frequency space.
* If `None`, it defaults to `f_cut_ratio`

**Effect:**

* Smaller → sharper cutoff (aggressive filtering)
* Larger → smoother transition

##### `window_type = "hann"`

* Applies a window function before FFT to reduce edge discontinuities.

**Effect:**

* Prevents artificial ringing in the filtered signal.
* Improves stability of FFT-based filtering.

##### `chunk_size = 128`

* Processes data in chunks along the spatial dimension.

**Effect:**

* Smaller chunks → lower memory usage, slower
* Larger chunks → faster, but higher memory requirement

##### `max_fft_bytes_safe`

* Memory threshold to decide FFT strategy

**Effect:**

* If estimated FFT size exceeds this → switches to safer (slower) method.
* Prevents memory crashes for large datasets.

#### **Important Notes**

* Over-filtering can suppress **real CME dynamics** and introduce ringing artifacts.
* Under-filtering leaves **flickering artifacts**, which degrade optical flow accuracy.
* This step is **critical for ASPIICS data quality before optical flow**.
---


### 8. Arrow Scaling - Optical Flow

```python
arrow_scale = 4
```

**Purpose:**
Controls vector length in overlay plots. For high-cadence datasets, the displacement between frames can be very small, because of which the vector arrows might be too small to be visually resolvable. Hence, increasing the arrow scale multiplies the vector arrow length for visual clarity.

---


### 9. Output Control - Optical Flow

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

## While DOFCAT was designed for CME kinematics using coronagraph data, it can be adapted to analyse large-scale motions in other areas of solar physics as well.
