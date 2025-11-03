# Hybrid Monocular Distance Estimation using CNN + Pinhole Residual

**Course:** COMP 3106 – Introduction to Artificial Intelligence  
**Student:** Mohamad Ghanem  
**Group:** 195 – Individual

---

## 1. Background and Objectives

Estimating object distance from a single RGB image is a fundamental perception task in computer vision. Depth from a monocular camera is ambiguous in theory, but real scenes provide consistent cues such as object scale in pixels, ground contact position, and class priors.  

The **KITTI dataset** provides per-object 2D boxes and 3D locations in camera coordinates, so the forward distance $Z$ can be used as a supervised training target.

**Objective:**  
Build a lightweight convolutional neural network (CNN) that predicts per-object distance $\hat{Z}$ from cropped detections and compare it to a physics baseline derived from the **pinhole camera model**. The goal is a fast, lightweight metric for real-time distance estimation where deep vision models can be computationally heavy.

---

## 2. Methods

### **Pipeline**
For each labeled object:
1. Read 2D bounding box $(x_1, y_1, x_2, y_2)$  
2. Crop the RGB image with small padding  
3. Resize to $128 \times 128$  
4. Use the label $Z$ from `location=(X, Y, Z)` as the regression target  

### **Model**
A CNN backbone with two independent heads:
- **Classification head:** Softmax over $\{Car, Pedestrian, Cyclist\}$
- **Regression head:** Predicts scalar residual $\widehat{\Delta Z}$

### **Hybrid Estimate**
The pinhole anchor uses focal length $f_x$ (from KITTI calibration) and class-average height $H_c$ (computed only on the training split to avoid leakage).  
With box height $h$ in pixels:

$$
Z_{\text{pin}} \approx \frac{f_x \cdot \bar{H}}{h}, \quad \text{where} \quad 
\bar{H} = \sum_{c} p_c \, H_c
$$

Final estimate:

$$
\hat{Z} = Z_{\text{pin}} + \widehat{\Delta Z}
$$

This combines a physics-based first guess with a learned correction for errors due to pose, occlusion, and camera pitch.

### **Loss Function**
Multi-task objective:

$$
\mathcal{L} = \text{CrossEntropy}(y_{\text{class}}, \hat{\mathbf{p}}) + 
\lambda \cdot \text{SmoothL1}(\hat{Z}, Z)
$$

$\lambda$ is tuned so both terms have similar scale.  
Predicting $\log Z$ or using Smooth L1 improves numerical stability.

---

## 3. Dataset and Environment

### **Dataset**
**KITTI Object Detection**:  
RGB images with labels containing type, 2D bounding box, 3D dimensions, and `location=(X, Y, Z)` in meters.  
Use objects of classes **Car**, **Pedestrian**, **Cyclist**.  
Ignore `DontCare` labels and optionally filter heavily truncated or occluded samples.

### **Targets and Calibration**
- Use $Z$ from `location` as the ground truth distance  
- Use $f_x$ from each calibration file  
- Compute class-average heights $H_c$ from the training split only  

### **Environment**
Python, PyTorch, OpenCV.  
A pretrained detector (e.g., **YOLOv8**) may be used to generate boxes for demo purposes.  
All training and evaluation use ground-truth boxes to focus on distance estimation.

---

## 4. Validation and Metrics

### **Splits**
Train, validation, and test object crops — ensuring no image overlap across splits.

### **Baselines**
- **Pinhole baseline:** $Z_{\text{pin}} = \dfrac{f_x \cdot H_c}{h}$
- **Plain CNN:** Directly predicts $\hat{Z}$ without $Z_{\text{pin}}$

### **Metrics**
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- $R^2$ on test set
- Per-class MAE
- Nearest-object accuracy (how often the model identifies the closest object correctly by ranking distances)

### **Visuals**
- Scatter plots of $\hat{Z}$ vs $Z$
- Error vs distance curves
- Example frames annotated with predicted $\hat{Z}$ and geometric $Z_{\text{pin}}$

---

## 5. Novelty

This project combines a [pinhole camera model](https://en.wikipedia.org/wiki/Pinhole_camera_model) with a lightweight CNN that learns residual corrections for per-object distance.  

Key contributions:
- Object-level scalar estimation (efficient and explainable)
- Probability-weighted class heights for smooth geometric priors
- Novel rank-based “nearest object” metric for practical evaluation

---

## 6. Weekly Schedule

| **Week** | **Tasks** |
|-----------|-----------|
| **1** | Research related papers. Parse KITTI labels and calibration. Build crop dataset with $Z$ targets. Compute class-average heights $H_c$. |
| **2** | Implement CNN backbone and two heads (plain approach). Train plain CNN to predict $\hat{Z}$. Set up logging and validation curves. |
| **3** | Add pinhole anchor and residual head (hybrid approach). Tune $\lambda$. Run ablations on crop-only vs crop + scalar cues. |
| **4** | Evaluate on test split. Compute MAE, RMSE, $R^2$, per-class results. Measure nearest-object accuracy. Produce plots. |
| **5** | Prepare demo and finalize report. Add optional experiments. |

---

## References

1. Eigen, D., Puhrsch, C., & Fergus, R. (2014). *Depth map prediction from a single image using a multi-scale deep network.* NeurIPS 27. [PDF](https://www.cs.toronto.edu/~bonner/courses/2022s/csc2547/papers/discriminative/image-transformation/depth-prediction,-eigen,-nips-2014.pdf)
2. Laina, I., Rupprecht, C., Belagiannis, V., Tombari, F., & Navab, N. (2016). *Deeper depth prediction with fully convolutional residual networks.* [DOI](https://doi.org/10.1109/3DV.2016.32)
3. Godard, C., Mac Aodha, O., & Brostow, G. (2017). *Unsupervised monocular depth estimation with left-right consistency.* [DOI](https://doi.org/10.1109/CVPR.2017.699)
4. Mousavian, A., Anguelov, D., Flynn, J., & Kosecka, J. (2017). *3D bounding box estimation using deep learning and geometry.* [DOI](https://doi.org/10.1109/CVPR.2017.597)
5. **KITTI Vision Benchmark Suite.** [Website](http://www.cvlibs.net/datasets/kitti/)