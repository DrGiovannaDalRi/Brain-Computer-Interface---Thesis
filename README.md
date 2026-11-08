# Brain-Computer-Interface---Thesis

## Author
Dr. Giovanna Dal Rì, Medical Doctor and Biomedical Engineer
giovanna.dalri.doctor@proton.me

## Motivation
This project was conducted for my thesis in Medicine and Biomedical Engineering (MedTec). I personally recorded the EEG data and entirely programmed the BCI system.

## Description
End-to-end BCI pipeline developed as an undergraduate (medicine and engineering) thesis project, bridging EEG signal acquisition, signal processing, machine learning decoding, and application as an interactive gaming platform.

The EEG data were retrieved in Aalborg University from two healthy people. The events are right hand movement, left hand movement and idle. No muscle movements were recorded.

### Signal processing
Signal processing involves:
- ...

### Feature extraction 
Feature extraction involves:
- ...

### Machine learning
Machine Learning involves:
- ...
The model with the highest accuracy was ----

### Performance 
The performances of the models were evaluated through ---

### Validation
The BCI system was validated in real-time by one of the two healthy people whose EEG data were initially recorded. The validation used an exergame platform that showed when and which hand to move.

### Results
The features were selected through ---
The kept features are: ---
The model with the highest accuracy is: ---  with --- %

## Tech Stack
* Language: Python 3.10+
* Signal Processing & ML: `NumPy`, `SciPy`, `Scikit-Learn`, `MNE-Python`
* Real-Time Streaming: `pylsl` (Lab Streaming Layer)
