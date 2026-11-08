<img width="792" height="147" alt="image" src="https://github.com/user-attachments/assets/aaf5d8b9-8678-4c97-b035-200e6488b3c7" /># Brain-Computer-Interface---Thesis

## Author
Dr. Giovanna Dal Rì, Medical Doctor and Biomedical Engineer
giovanna.dalri.doctor@proton.me

## Motivation
This project was conducted for my thesis in Medicine and Biomedical Engineering (MedTec). I personally recorded the EEG data and entirely programmed the BCI system.

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

### Features augmentation
The features were augmented through WGAN-GP algorithm. This improved the performance of the classification.

### Machine learning
Different Machine Learning models were tested and only the one with the highest accuracy was kept, Random Forest.
A cascaded classifier framework was implemented. The first classifier distinguishes between imaginary movement and idle, while the second distinguishes between right-hand and left-hand imaginary movement.
The best combination of hyperparameters was found through the grid search.

<img width="860" height="911" alt="image" src="https://github.com/user-attachments/assets/9b862fec-89c0-4cc2-962c-f553e26933f3" />

### Validation
The BCI system was validated in real-time by one of the two healthy people whose EEG data were initially recorded. The validation used an exergame platform that showed when and which hand to move.


### Performance 
The performances of the models were evaluated through ---



### Results
The features were selected through ---
The kept features are: ---
The model with the highest accuracy is: ---  with --- %

## Tech Stack
* Language: Python 3.10+
* Signal Processing & ML: `NumPy`, `SciPy`, `Scikit-Learn`, `MNE-Python`
* Real-Time Streaming: `pylsl` (Lab Streaming Layer)
