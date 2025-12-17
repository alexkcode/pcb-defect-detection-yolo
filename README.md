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

1. Kim, J., Ko, J., Choi, H., & Kim, H. (2021). **Printed Circuit Board Defect Detection Using Deep Learning via A Skip-Connected Convolutional Autoencoder**. *Sensors*, 21(15), 4968.  
   [https://doi.org/10.3390/s21154968](https://www.mdpi.com/1424-8220/21/15/4968)

2. Huang, W., & Wei, P. (2019). **A PCB Dataset for Defects Detection and Classification**. *arXiv preprint arXiv:1901.08204*.  
   [https://arxiv.org/abs/1901.08204](https://arxiv.org/abs/1901.08204)

3. Ultralytics. (2024). **Ultralytics YOLO11 - Ultralytics YOLO Docs**.  
   [https://docs.ultralytics.com/models/yolo11/](https://docs.ultralytics.com/models/yolo11/)  
   *Accessed: 2024-12-05*

4. Ultralytics. (2025). **YOLO11 vs YOLOv8: Architectural Evolution and Performance Analysis**.  
   [https://docs.ultralytics.com/compare/yolo11-vs-yolov8/](https://docs.ultralytics.com/compare/yolo11-vs-yolov8/)  
   *Accessed: 2024-12-05*

5. NVIDIA. (2023). **Train With Mixed Precision - NVIDIA Docs**.  
   [https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)  
   *Accessed: 2024-12-05*

6. TensorFlow. (2024). **GPU Memory Growth Configuration**.  
   [https://www.tensorflow.org/guide/gpu](https://www.tensorflow.org/guide/gpu)  
   *Accessed: 2024-12-05*

7. PyTorch. (2024). **Memory Optimization Techniques for Efficient Deep Learning**.  
   [https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)  
   *Accessed: 2024-12-05*

8. OpenCV. (2024). **Image Preprocessing and Enhancement Techniques**.  
   [https://docs.opencv.org/4.x/](https://docs.opencv.org/4.x/)  
   *Accessed: 2024-12-05*

9. Scikit-learn. (2024). **Model Evaluation Metrics**.  
   [https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)  
   *Accessed: 2024-12-05*

10. Google Colab. (2024). **Using GPUs and TPUs**.  
    [https://colab.research.google.com/notebooks/gpu.ipynb](https://colab.research.google.com/notebooks/gpu.ipynb)  
    *Accessed: 2024-12-05*

11. Roboflow. (2024). **PCB Defect Detection Datasets**.  
    [https://universe.roboflow.com/](https://universe.roboflow.com/)  
    *Accessed: 2024-12-05*

12. GeeksforGeeks. (2025). **Denoising AutoEncoders In Machine Learning**.  
    [https://www.geeksforgeeks.org/machine-learning/denoising-autoencoders-in-machine-learning/](https://www.geeksforgeeks.org/machine-learning/denoising-autoencoders-in-machine-learning/)  
    *Accessed: 2024-12-05*

13. Medium. (2024). **Gradient Accumulation in PyTorch**.  
    [https://medium.com/biased-algorithms/gradient-accumulation-in-pytorch-36962825fa44](https://medium.com/biased-algorithms/gradient-accumulation-in-pytorch-36962825fa44)  
    *Accessed: 2024-12-05*

14. Visionular. (2024). **Making Sense of PSNR, SSIM, VMAF**.  
    [https://visionular.ai/vmaf-ssim-psnr-quality-metrics/](https://visionular.ai/vmaf-ssim-psnr-quality-metrics/)  
    *Accessed: 2024-12-05*

15. Pandey, A. K. (2024, March). **Structural Similarity Index (SSIM)**.  
    [https://medium.com/@akp83540/structural-similarity-index-ssim-c5862bb2b520](https://medium.com/@akp83540/structural-similarity-index-ssim-c5862bb2b520)  
    *Accessed: 2024-12-05*

