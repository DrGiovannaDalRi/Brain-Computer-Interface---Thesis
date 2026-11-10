# Brain-Computer-Interface and Neurorecovery---Thesis

## Author
Dr. Giovanna Dal Rì, Medical Doctor and Biomedical Engineer

giovanna.dalri.doctor@proton.me

## Motivation
This project was conducted for my thesis in Medicine and Biomedical Engineering (MedTec). I personally recorded the EEG data and entirely programmed the BCI system.

## Pipeline
1. EEG Recording
2. EEG Filtering
3. Epoch Extraction
4. Blink Rejection
5. Large Laplacian
6. Matched Filtering
7. Feature Extraction
8. WGAN-GP Feature Augmentation
9. Random Forest Classification
10. SHAP Explainability

## Description
End-to-end BCI pipeline developed as an undergraduate (medicine and engineering) thesis project, bridging EEG signal acquisition, signal processing, machine learning decoding, and application as an interactive gaming platform.

The EEG data were retrieved in Aalborg University from two healthy people. The used scalp electrodes are: FP1, F3, F4, C3, CZ, C4, P3, P4. The events are right hand movement, left hand movement and idle. No muscle movements were recorded.

### Signal processing
Signal processing involves:
- Detrending filtering: removes trends from data so that underlying patterns or fluctuations can be analyzed without long-term influences distorting the results
- Bandpass filtering: keeps EEG data in the range 8Hz – 30Hz, including Alpha and Beta waves, related to motor
- Epoching: divides the continuous EEG data into windowed EEG data, each containing one event of interest
- Blink and artifacts detection: exploits the FP1 electrode to detect blinks or other artifacts in the EEG signal
- Large Laplacian filtering: enhances the edges and detects the fine details, using the following formula
- Matched filtering: correlates the EEG signal with a predefined template designed to resemble the expected signal

### Feature extraction 
The extracted features are temporal and spectral. Only the most influential features according to their SHAP values are kept. 

The temporal features of interest are:
- Point of maximum negativity
- Mean amplitude
- Windowed mean amplitude
- Linear regression slope
- Linear regression intercept
- Variance
- Skewness
- Kurtosis

The spectral features of interest are:
- PSD for every 1Hz bin in range of 8Hz-30Hz 

### Features augmentation
The features were augmented through WGAN-GP algorithm. This improved the performance of the classification.

### Machine learning
Different Machine Learning models were tested and only the one with the highest accuracy was kept, Random Forest.
A cascaded classifier framework was implemented. The first classifier distinguishes between imaginary movement and idle, while the second distinguishes between right-hand and left-hand imaginary movement.
The best combination of hyperparameters was found through the grid search.

<div align="center">
<img width="400" height="450" alt="Cascaded Framework" src="https://github.com/user-attachments/assets/9b862fec-89c0-4cc2-962c-f553e26933f3" />
</div>

### Actuator 
The actuator was an exergame platform. It gave different commands (right-hand, left-hand or idle) and compared it with the execution. If the execution was correct, the game proceeds.

<div align="center">
<img width="600" height="300" alt="Actuator" src="https://github.com/user-attachments/assets/bdf15c66-5507-4c60-a3ab-90fc97f142b2" />
</div>

### Validation
The BCI system was validated in real-time by one of the two healthy people whose EEG data were initially recorded. The validation used a task-reward system built in the actuator platform.

<div align="center">
<img width="600" height="400" alt="Validation Flowchart" src="https://github.com/user-attachments/assets/c9de0c09-3b1a-4076-a3eb-41f1b3efa887" />
</div>

### Performance 
The performance of the feature augmentation with WGAN-GP algorithm was evaluated through Pearson Correlation scores and KS Test scores.

The performance of the ML models was evaluated through 5-Fold Cross Validation.

### Results
Feature augmentation reported the following Pearson Correlation scores for each class:
- Idle (0.91)
- Right-wrist movement (0.89)
- Left-wrist movement (0.76)

The following KS Test scores were also reported for each class: 
- Idle (0.09)
- Right-wrist movement (0.12)
- Left-wrist movement (0.18)
  
The first ML classifier presented the following performance values:
- Accuracy = 0.933 (± 0.010)
- Precision: 0.95 ± 0.01
- Recall: 0.95 ± 0.01 
- F1-Score: 0.95 ± 0.01

The confusion matrix for the first classifier is reported in the following image:
<div align="center">
<img width="500" height="350" alt="Confusion Matrix First Classifier" src="https://github.com/user-attachments/assets/f46f6e68-0e6c-4b35-b2c6-1ef55f9b5929" />
</div>

The SHAP explainability regarding the first classifier is shown in the following image:
<div align="center">
<img width="600" height="350" alt="SHAP explainability First Classifier" src="https://github.com/user-attachments/assets/9aaa40d6-d70c-462c-8133-92c44a283e38" />
</div>

The most influent EEG feature was Variance.

The second ML classifier presented the following performance values:
- Accuracy = 0.965 (± 0.010)
- Precision: 0.97 ± 0.01
- Recall: 0.96 ± 0.01 
- F1-Score: 0.97 ± 0.01

The confusion matrix for the second classifier is reported in the following image:
<div align="center">
<img width="500" height="350" alt="Confusion Matrix Second Classifier" src="https://github.com/user-attachments/assets/ad058e18-208c-4283-a358-87a448cc5346" />
</div>

The SHAP explainability regarding the second classifier is shown in the following image:
<div align="center">
<img width="600" height="350" alt="SHAP explainability Second Classifier" src="https://github.com/user-attachments/assets/698848cd-fb1f-4190-9f67-e86e0e506ae5" />
</div>

The most influent EEG feature was PSD between 29Hz and 30Hz.

### Discussion
The presented BCI system reports fundamental characteristics that distinguish it from other systems. Firstly, no physical movement was needed for the BCI functioning. Secondly, the task-reward framework implemented in the actuator allowed validation and possibly induction of neuroplasicity and neurorecovery.

### Conclusion
The system can be used without limitation of pathologies (besides the limitation to the comprehension of the movements or activities).  

## Tech Stack
* Language: Python 3.10+

## License
This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

Under this license, you are free to use, modify, and distribute the code for academic or commercial purposes, provided that original authorship is attributed and modified files retain notices of changes.

## Citation
If you use this code, the WGAN-GP feature augmentation pipeline, or the BCI framework in your research, please cite it as follows:

```bibtex
@masterthesis{DalRi2025BCI,
  author       = {Dr. Giovanna Dal Rì, MD, BME},
  title        = {End-to-End BCI Pipeline with WGAN-GP Feature Augmentation and Exergame Actuator for Neurorecovery},
  school       = {Medicine and Biomedical Engineering (MedTec), Humanitas University and Politecnico di Milano},
  year         = {2025},
  note         = {GitHub Repository: [https://github.com/DrGiovannaDalRi/Brain-Computer-Interface---Thesis]
}
