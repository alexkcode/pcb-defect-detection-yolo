# Enhanced PCB Defect Detection Using Skip-Connected Autoencoder Preprocessing and YOLOv11 Object Detection

This repository presents an advanced system for **PCB defect detection** that combines a **skip-connected convolutional autoencoder** for image preprocessing with **YOLOv11** for object detection. The autoencoder performs denoising and feature enhancement on PCB images before feeding them into the YOLOv11 detector, aiming to improve overall detection accuracy.

## Abstract

Printed Circuit Board (PCB) defect detection is crucial for ensuring high-quality electronic manufacturing. Traditional machine vision methods often struggle with small defects and variable noise conditions. This project implements a deep learning pipeline that uses a skip-connected autoencoder—based on the architecture from Kim et al. (MDPI Sensors, 2021)—to denoise and enhance input images prior to detection using **YOLOv11**.

Key features of the system include:
- **Autoencoder-based preprocessing** for noise reduction and feature preservation
- Integration with **YOLOv11n (nano)** for fast and accurate defect localization
- **Memory optimization techniques** such as mixed precision training, gradient accumulation, and progressive image sizing to enable training on resource-limited environments (e.g., Google Colab)

Experimental results on the Huang and Wei PCB defect dataset show:
- **PSNR improvement of 7.2 dB**, indicating effective denoising
- **SSIM score of 0.750**, confirming structural integrity preservation
- **97.1% average precision** with a **2.1% false positive rate**
- **No improvement in mAP50 (0.0%)** when comparing baseline YOLOv11 vs. autoencoder-enhanced version, suggesting YOLOv11's inherent robustness may reduce the added value of preprocessing in this setup

## Methodology

### System Architecture
The detection pipeline consists of two stages:
1. **Skip-connected autoencoder** for image denoising and enhancement
2. **YOLOv11 object detector** for localizing six common PCB defects:
   - Missing hole
   - Mouse bite
   - Open circuit
   - Short circuit
   - Spur
   - Spurious copper

![System Architecture](figures/system_architecture.png)

### Autoencoder Design
Based on Kim et al.'s architecture:
- **Encoder**: Four convolutional layers with filters `[64, 64, 128, 128]` and kernel sizes `[(5,5), (5,5), (3,3), (3,3)]`, followed by max pooling
- **Decoder**: Symmetric upsampling layers with **skip connections** that concatenate encoder and decoder feature maps
- Skip connection operation:  
  $f(E_i, D_i) = E_i \oplus D_i$  
  where $\oplus$ denotes element-wise addition

![Autoencoder Architecture](figures/autoencoder_architecture.png)

### YOLOv11 Integration
- Uses **YOLOv11n** with pretrained COCO weights
- Input size: $640 \times 640$
- Leverages optimized backbone architectures (**C3k2**, **C2PSA**) for efficient multi-scale feature extraction

### Memory Optimization Techniques
To support training on limited hardware:
- **Mixed precision (FP16)**: Reduces memory usage by ~50%
- **Gradient accumulation**: Simulates larger batch sizes (effective batch = micro-batch × accumulation steps)
- **Progressive image sizing**: Trains autoencoder on $256 \times 256$ instead of $400 \times 400$, reducing memory by 59%

## Dataset
- Uses the **PCB defect dataset by Huang and Wei** (2019)
- Contains 693 images with 2,953 labeled defects across six categories
- Augmented with geometric transformations and Gaussian noise to address class imbalance

![Sample Defects](figures/sample_defects.png)

## Training Configuration

| Component        | Settings |
|------------------|--------|
| **Autoencoder**  | SGD optimizer, momentum 0.9, weight decay $5 \times 10^{-4}$, step LR schedule (0.1 → 0.0008), 300 epochs, batch size 128 (via gradient accumulation) |
| **YOLOv11**      | Batch size 16, initial LR 0.01, 100 epochs, standard YOLO augmentations adapted for PCBs |

## Evaluation Metrics

### Detection Performance
- **mAP50**: Mean Average Precision at IoU threshold 0.5
- **mAP50–95**: Average AP across IoU thresholds from 0.5 to 0.95
- **Precision, Recall, F1-Score**

### Image Quality (Autoencoder)
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures reconstruction quality
- **SSIM (Structural Similarity Index Measure)**: Evaluates structural similarity between original and reconstructed images

## Results

### Autoencoder Denoising Performance
- **Average PSNR improvement**: 7.2 ± 0.6 dB
- **SSIM score**: 0.750 ± 0.028

![Reconstruction Results](figures/reconstruction_results.png)

### Detection Performance Comparison

| Model             | mAP50 | mAP50–95 | F1-Score |
|-------------------|-------|----------|---------|
| Baseline YOLOv11  | 0.972 | 0.546    | 0.973   |
| Enhanced YOLOv11  | 0.972 | 0.546    | 0.973   |
| **Improvement**   | +0.000 | +0.000   | +0.000  |

> 📊 *No improvement observed in detection metrics despite significant denoising gains.*

## Discussion

- The **skip connections** in the autoencoder help preserve fine details critical for detecting small PCB defects.
- Despite strong **denoising performance (7.2 dB PSNR gain)**, there is **no measurable improvement in detection accuracy**, likely because:
  - **YOLOv11 is already highly robust** to noise and variations
  - Preprocessing may inadvertently remove subtle defect cues
  - Optimization goals of autoencoder (reconstruction) and detector (localization) may conflict
- Memory optimizations enabled feasible training on consumer-grade hardware but may have limited model capacity

### Limitations
- Reliance on synthetic or limited real-world defect data
- Performance may degrade on unseen defect types
- No significant gain from preprocessing when using state-of-the-art detectors

## Conclusion

This work demonstrates a complete pipeline for PCB defect detection using **autoencoder preprocessing and YOLOv11**. While the autoencoder effectively denoises PCB images (PSNR +7.2 dB), it does **not improve detection performance** over baseline YOLOv11, suggesting that modern detectors like YOLOv11 are already robust enough to handle raw, noisy inputs.

Future directions:
- Explore alternative filtering or enhancement methods
- Test on diverse real-world PCB datasets
- Optimize end-to-end training of both components
- Integrate into industrial inspection systems

## Acknowledgments

We thank the open-source community for providing datasets and tools. Special thanks to **Kim et al.** for their foundational work on skip-connected autoencoders for PCB defect detection, and to **Professor Ahmet Enis Cetin** for guidance on image processing fundamentals.

## References
