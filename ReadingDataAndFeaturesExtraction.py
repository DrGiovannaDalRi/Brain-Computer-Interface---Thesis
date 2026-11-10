###EEG SIGNAL PROCESSING AND FEATURE EXTRACTION
##Signal processing of raw EEG data recorded from two healthy patients. The events are right-hand imaginary movement, left-hand imaginary movement and idle.
##After signal processing, temporal and spatial features are extracted.

"""
Pipeline:

1. Load raw EEG recordings acquired with OpenBCI Cyton
2. Convert recordings into MNE RawArray objects
3. Apply signal preprocessing:
   - Detrending
   - Butterworth band-pass filtering (8–30 Hz)
4. Segment recordings into 4-second epochs
5. Separate epochs into:
   - Right-hand motor imagery
   - Left-hand motor imagery
   - Idle state
6. Detect eye blinks using the FP1 channel
7. Remove blink-contaminated epochs
8. Balance the number of epochs across classes
9. Remove FP1 channel after artifact rejection
10. Apply spatial processing:
    - Large Laplacian Spatial Filtering
    - Matched Filtering
11. Save processed epochs as FIF files
12. Extract handcrafted EEG features:
    - Temporal features
    - Spectral features
    - Entropy-based features
    - Time-scale features
13. Save extracted features for machine-learning analysis

Output:
- Processed EEG epochs (.fif)
- Processed EEG datasets (.xlsx)
- Feature matrices (.xlsx)

%Number of channels = 8
%Sample Rate = 250 Hz
%Board = OpenBCI_GUI$BoardCytonSerial
"""


##IMPORT LIBRARIES

#Libraries needed for the handling and plotting of the data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Libraries needed for the signal processing and machine learning
import mne
from scipy.signal import correlate

#Libraries needed for the feature extraction
from mne_features.univariate import compute_app_entropy, compute_samp_entropy, compute_pow_freq_bands, compute_variance, compute_mean, compute_kurtosis, compute_skewness, compute_katz_fd, compute_wavelet_coef_energy

##DEFINE THE VARIABLES

#Define the sampling frequency
samplingFreq = 250

#Generate the arrays for the channels names
channelsNames = ['FP1', 'F3', 'F4', 'C3', 'Cz', 'C4', 'P3', 'P4']

#Define the frequencies for the bandpass filter in signal processing
LOWCUT = 8  # Lower cut-off frequency in Hz
HIGHCUT = 30  # Higher cut-off frequency in Hz
FORDER = 5  # Filter order

#Define the window size and step size for the sliding window
windowSize = 4  # in seconds
stepSize = 10  # in seconds
numMovementsInitial = 51

#Define the threshold for blink detection
BLINKTHRESHOLD = 150 #in microvolts

#Define the baseline start and stop timing
BASELINESTART = 0
BASELINESTOP = 1.8

#EXPLANATION OF DATA
#The EEG data contain hand movements every 10 seconds
#The first movement was performed by the right hand, while the second movement by the left hand

#DEFINITION OF FILES TO USE
#Define the path of the recordings
RecordingsPath = r"C:\Users\--\Recordings"
#Define the path to the file with 70 hand movements
HandMovements70 = r"C:\Users\--\HandMovements70.txt"
#Define the path to the file idle
Idle = r"C:\Users\--\Idle.txt"
##Define the path to the file with 32 hand movements
HandMovements32 = r"C:\Users\--\HandMovements32.txt"

##PREPARE THE DATA
class PrepareData:

    def InitializeData(self):

        #Read the data from the txt file as a csv file
        #The range goes from 1 to 9 to keep the first column out (which is not data from a channel)
        dataHandMovements70 = pd.read_csv(HandMovements70, skiprows= 4,usecols=range(1,9), delimiter=',', engine='python')
        dataIdle = pd.read_csv(Idle, skiprows= 4,usecols=range(1,9), delimiter=',', engine='python')
        dataHandMovements32 = pd.read_csv(HandMovements32, skiprows= 4,usecols=range(1,9), delimiter=',', engine='python')

        dataInitializedHandMovements70 = self.AdjustData(dataHandMovements70)
        dataInitializedHandMovements32 = self.AdjustData(dataHandMovements32)
        dataInitializedIdle = self.AdjustData(dataIdle)

        return dataInitializedHandMovements70, dataInitializedHandMovements32, dataInitializedIdle

    def AdjustData(self, data):

        # Multiply the data by -1, since the data were obtained from the bottom row of the bioamplifier
        data = data * -1
        # Transpose the data: (n_times, n_chans) -> (n_chans, n_times)
        data = data.T

        return data

##GENERATION OF MNE RAW ARRAY
class GenerateMNEArrays:

    def GenerateRawArrayObject(self, dataHandMovements70, dataHandMovements32, dataIdle):

        # Define the type of the channels
        channelsTypes = ['eeg'] * len(channelsNames)

        # Generate an info object
        infoMNE = mne.create_info(ch_names=channelsNames, sfreq=samplingFreq, ch_types=channelsTypes)

        # Generate an RawArray object
        rawHandMovements70 = mne.io.RawArray(dataHandMovements70, infoMNE)
        rawIdle = mne.io.RawArray(dataIdle, infoMNE)
        rawHandMovements32 = mne.io.RawArray(dataHandMovements32, infoMNE)

        return infoMNE, rawHandMovements70, rawHandMovements32, rawIdle

##FREQUENCY SIGNAL PROCESSING
class FrequencySignalProcessing:

    def SignalProcessing(self, data, infoMNE):
        # Extract data from the raw object
        data = data.get_data()

        #Call the function to apply frequency filtering
        filteredData=self.FrequencyFiltering(data)

        #Generate again the RawArray object for the division in windows
        filteredData = mne.io.RawArray(filteredData, infoMNE)

        return filteredData

    def FrequencyFiltering(self, data):
        # Create a copy of the data to avoid modifying the original
        data = data.copy()

        # Apply detrending to everything except for the data from FP1 (the first channel)
        data[1:, :] = mne.filter.detrend(data[1:, :], order=1)

        # Apply a bandpass filter between the selected frequencies to everything except for the data from FP1 (the first channel)
        #The lowcut and highcut frequencies are included in the selected range
        data[1:, :] = mne.filter.filter_data(data[1:, :], samplingFreq, LOWCUT, HIGHCUT, method='iir', iir_params=dict(order=FORDER, ftype='butter'))

        return data

#GENERATE THE EPOCHS
class DivideWindows:

    # Divide the data in windows of 4 seconds: the hand movements happen every 10 seconds
    def DivideWindows(self, infoMNE, FilteredHandMovements70, FilteredHandMovements32, FilteredIdle):

        #Divide the data in windows of 4 seconds
        epochsRight70, epochsLeft70 = self.IterateOverDataMovement(FilteredHandMovements70)
        epochsRight32, epochsLeft32 = self.IterateOverDataMovement(FilteredHandMovements32)
        windowIdle = self.IterateOverDataIdle(FilteredIdle)

        #Convert the epochs in the form of MNE EpochsArray
        #Apply baseline correction (the interval for the epoch goes from [0.0, 3.996], so the baseline is considered the 2 seconds before the event)
        epochsRight70 = mne.EpochsArray(epochsRight70, infoMNE).apply_baseline(baseline=(0, 2))
        epochsLeft70 = mne.EpochsArray(epochsLeft70, infoMNE).apply_baseline(baseline=(0, 2))
        epochsRight32 = mne.EpochsArray(epochsRight32, infoMNE).apply_baseline(baseline=(0, 2))
        epochsLeft32 = mne.EpochsArray(epochsLeft32, infoMNE).apply_baseline(baseline=(0, 2))
        windowIdle = mne.EpochsArray(windowIdle, infoMNE).apply_baseline(baseline=(0, 2))

        #Concatenate the epochs
        epochsRight = mne.concatenate_epochs([epochsRight70, epochsRight32])
        epochsLeft = mne.concatenate_epochs([epochsLeft70, epochsLeft32])

        #Detect the blinks occurring during the epochs and remove the respective windows
        epochsRight, detectedBlinksRight = self.BlinkDetector(epochsRight)
        epochsLeft, detectedBlinksLeft = self.BlinkDetector(epochsLeft)
        windowIdle, detectedBlinksIdle = self.BlinkDetector(windowIdle)
        print(f"The total number of detected blinks in RIGHT is:{detectedBlinksRight}")
        print(f"The total number of detected blinks in LEFT is:{detectedBlinksLeft}")
        print(f"The total number of detected blinks in IDLE is:{detectedBlinksIdle}")

        #Define the total # of blinks detected
        totalDetectedBlinks = detectedBlinksRight + detectedBlinksLeft + detectedBlinksIdle
        print(f"The total number of detected blinks is:{totalDetectedBlinks}")

        #Obtain the same # of windows in each group of epochs
        epochsRight, epochsLeft, windowIdle = self.AlignEpochs(epochsRight, epochsLeft, windowIdle)

        epochsRight, epochsLeft, windowIdle, infoMNE = self.RemoveFP1(epochsRight, epochsLeft, windowIdle)

        return epochsRight, epochsLeft, windowIdle, infoMNE

    def IterateOverDataMovement(self, data):

        # Calculate the number of samples in each window
        samplesPerWindow = int(windowSize * samplingFreq)

        # Calculate the number of samples to skip for each window
        samplesToSkip = int(stepSize * samplingFreq)

        # Extract the first window separately
        # Due to the fact that it is the first window, it starts after 8 seconds, not 6 like the others
        startIdxFirstWindow = int(8 * samplingFreq)
        endIdxFirstWindow = startIdxFirstWindow + samplesPerWindow

        firstWindowEpoch = data[:, startIdxFirstWindow:endIdxFirstWindow][0]

        # Initialize lists to store the segmented data
        rightSegmentedData = []
        leftSegmentedData = []

        # Add the first window to the appropriate list
        rightSegmentedData.append(firstWindowEpoch)

        # Iterate over the data to extract windows
        # Epochs will contain all the 4-seconds windows of data (meaning the epochs)
        for i in range(startIdxFirstWindow + samplesToSkip, len(data.times) - samplesPerWindow, samplesToSkip):
            startIdx = i #define the startIdx, needed to extract the window, corresponding to the # of the window
            endIdx = i + samplesPerWindow

            #After slicing, putting [0] selects the first element from the resulting array. The assumption here is that data is a 2D array
            newEpoch = data[:, startIdx:endIdx][0]  # Extract the windowed data

            #Check if the window index is even or odd and add to the respective list
            #The division of the windows according to right or left is based on the # of windows in each list: all windows should be inserted here
            #It must be done before blink detection
            if (len(rightSegmentedData) + len(leftSegmentedData)) % 2 == 0:
                rightSegmentedData.append(newEpoch)
            else:
                leftSegmentedData.append(newEpoch)

        return rightSegmentedData, leftSegmentedData

    def IterateOverDataIdle(self, data):
        #Define the number of maximum windows to obtain, considering that some windows could have a blink in them, so adding some more for reserve
        maximumWindows = numMovementsInitial + 10

        # Calculate the number of samples in each window
        samplesPerWindow = int(windowSize * samplingFreq)

        # Calculate the number of samples to skip for each window
        samplesToSkip = int(len(data.times) / maximumWindows)
        # Initialize a list to store the segmented data
        epochs = []

        # Iterate over the data to extract windows
        # Epochs will contain all the 4-seconds windows of data
        for i in range(0, len(data.times) - samplesPerWindow, samplesToSkip):
            startIdx = i
            endIdx = i + samplesPerWindow
            newEpoch = data[:, startIdx:endIdx][0]  # Extract the windowed data
            epochs.append(newEpoch)

        return epochs

    #CHECK EVERY EPOCH: IF THE FP1 IN THAT WINDOW > THRESHOLD --> THERE IS A BLINK IN THAT WINDOW
    def BlinkDetector(self, epochs):
        #Initialize the list to store windows without blinks
        filteredWindows = []
        #Initialize the variable to store the number of blinks detected
        detectedBlinks = 0

        #Get the data from the MNE array
        data = epochs.get_data(copy = False)

        for epoch_index in range(data.shape[0]):
            fp1_data = data[epoch_index, 0, :]
            #Remove any artifact above BLINKDETECTION threshold, even in the case in which the blink is recorded negative
            if np.any(np.abs(fp1_data) > BLINKTHRESHOLD):
                detectedBlinks += 1
            else:
                filteredWindows.append(data[epoch_index])

        #Return error in the case in which all the epochs are eliminated (due to blinks and/or wrong recording)
        if len(filteredWindows) == 0:
            raise ValueError(
                "No more epochs - all were eliminated"
            )

        filteredWindows = mne.EpochsArray(np.array(filteredWindows), epochs.info)

        return filteredWindows, detectedBlinks

    def AlignEpochs(self, epochs1, epochs2, epochs3):

        # Get the number of epochs in each list
        numEpochs1 = len(epochs1.events)
        numEpochs2 = len(epochs2.events)
        numEpochs3 = len(epochs3.events)

        # Find the minimum number of epochs among the three lists
        minNumEpochs = min(numEpochs1, numEpochs2, numEpochs3)

        # Drop excess epochs to make each list have the same number of epochs
        epochs1 = self.dropExcessEpochs(epochs1, minNumEpochs)
        epochs2 = self.dropExcessEpochs(epochs2, minNumEpochs)
        epochs3 = self.dropExcessEpochs(epochs3, minNumEpochs)

        # Now all three epochs arrays should have the same number of epochs
        print(f"Epochs1 -after alignment- has {len(epochs1.events)} epochs")
        print(f"Epochs2 -after alignment- has {len(epochs2.events)} epochs")
        print(f"Epochs3 -after alignment- has {len(epochs3.events)} epochs")

        return epochs1, epochs2, epochs3

    #Function to drop excess epochs
    def dropExcessEpochs(self, epochs, target_num_epochs):
        num_epochs = len(epochs.events)
        if num_epochs > target_num_epochs:
            num_to_drop = num_epochs - target_num_epochs

            #Use the random function to choose a random epoch to drop, avoiding biases of choosing the first or last windows
            drop_indices = np.random.choice(range(num_epochs), num_to_drop, replace=False)
            epochs.drop(drop_indices)

        return epochs

    def RemoveFP1(self, epochsRight, epochsLeft, windowIdle):
        #Drop the FP1 data (no more needed after blink detection)
        channelsToDrop = ['FP1']

        epochsRight = epochsRight.drop_channels(channelsToDrop)
        epochsLeft = epochsLeft.drop_channels(channelsToDrop)
        windowIdle = windowIdle.drop_channels(channelsToDrop)

        #Generate new infoMNE, removing FP1 data
        newChannelsNames = channelsNames[1:]

        channelsTypes = ['eeg'] * len(newChannelsNames)

        infoMNE = mne.create_info(ch_names=newChannelsNames, sfreq=samplingFreq, ch_types=channelsTypes)

        return epochsRight, epochsLeft, windowIdle, infoMNE

###SPATIAL SIGNAL PROCESSING
class SpatialSignalProcessing():
    def Initialization(self, epochsRight, epochsLeft, windowIdle, infoMNE):

        # Apply spatial filtering to the epochs
        epochsRight = self.SpatialFiltering(epochsRight, infoMNE)
        epochsLeft = self.SpatialFiltering(epochsLeft, infoMNE)
        windowIdle = self.SpatialFiltering(windowIdle, infoMNE)

        # Get the template (average of the channel averages)
        templateRight = self.AveragedData(epochsRight)
        templateLeft = self.AveragedData(epochsLeft)
        templateIdle = self.AveragedData(windowIdle)

        # Apply matched filtering to the epochs
        epochsRight = self.MatchedFiltering(epochsRight, templateRight, infoMNE)
        epochsLeft = self.MatchedFiltering(epochsLeft, templateLeft, infoMNE)
        windowIdle = self.MatchedFiltering(windowIdle, templateIdle, infoMNE)  # Apply matched filtering to idle data

        return epochsRight, epochsLeft, windowIdle

    # LARGE LAPLACIAN SPATIAL FILTER LLSF
    def SpatialFiltering(self, data, infoMNE):
        # Formula of the filter: Cz - (F3 + F4 + C3 + C4 + P3 + P4) / 6

        data=data.get_data(copy=False)

        # Index for Cz channel
        Cz_idx = 3  #Index based on the channel order (since FP1 has been removed)
        other_channels_idx = [0, 1, 2, 4, 5, 6]  # Indices for F3, F4, C3, C4, P3, P4

        # Create filtered_data as a copy of the original data
        filtered_data = data.copy()

        # Apply the filter to each epoch
        for i in range(data.shape[0]):  # Iterate over epochs
            Cz = data[i, Cz_idx, :]
            mean_other_channels = np.mean(data[i, other_channels_idx, :], axis=0) # = (F3 + F4 + C3 + C4 + P3 + P4) / 6
            filtered_data[i, Cz_idx, :] = Cz - mean_other_channels

        filtered_epochs = mne.EpochsArray(filtered_data, infoMNE)

        return filtered_epochs

    def AveragedData(self, data):

        # Get the data from the epochs object
        data = data.get_data(copy=False)

        # Initialize a list to store the average of each channel
        channel_averages = []

        #Loop through each channel (7 channels)
        for i in range(7):
            # Extract data for the current channel across all epochs
            channel_data = [epoch[int(i)] for epoch in data]

            # Calculate the average for the current channel
            channel_average = np.mean(channel_data, axis=0)

            # Append the channel average to the list
            channel_averages.append(channel_average)

        # Calculate the overall average of the channel averages
        channel_averages=np.array(channel_averages)

        return channel_averages

    def MatchedFiltering(self, data, template, infoMNE):
        # Get the data from the epochs object
        data = data.get_data(copy=False)

        # Initialize the array for filtered data
        filtered_data = np.zeros_like(data)

        # Apply matched filtering for each epoch
        for i in range(data.shape[0]):  # For each epoch
            # Create a template from all epochs except the current one
            leave_one_out_data = np.delete(data, i, axis=0)

            # Calculate the template (average across epochs for each channel)
            template = np.mean(leave_one_out_data, axis=0)

            # Apply matched filtering for each channel
            for c in range(data.shape[1]):
                filtered_data[i, c] = correlate(data[i, c], template[c], mode='same')

        # Create a new Epochs object with the filtered data
        filtered_epochs = mne.EpochsArray(filtered_data, infoMNE)

        return filtered_epochs

#SAVE THE FILES AS .FIF
class SaveFiles():
    def SaveFiles(self, epochsRight, epochsLeft, windowidle):
        #Save the MNE array as a FIF file
        self.SaveFIFArray(epochsRight, epochsLeft, windowidle)

        #Save the files as dataframes
        self.ConvertDataframe(epochsRight, epochsLeft, windowidle)

    def SaveFIFArray(self, epochsRight, epochsLeft, idle):
        #Save the epochs to a FIF file
        epochsRight.save('epochsRight-epo.fif', overwrite=True)
        epochsLeft.save('epochsLeft-epo.fif', overwrite=True)
        idle.save('idle-epo.fif', overwrite=True)

    def ConvertDataframe(self, epochsRight, epochsLeft, windowIdle):
        # Convert epochs and idle to dataframes
        df_epochsRight = epochsRight.to_data_frame()
        df_epochsLeft = epochsLeft.to_data_frame()
        df_windowIdle = windowIdle.to_data_frame()

        # Save dataframes to Excel files
        df_epochsRight.to_excel('epochsRight.xlsx', index=False)
        df_epochsLeft.to_excel('epochsLeft.xlsx', index=False)
        df_windowIdle.to_excel('windowIdle.xlsx', index=False)

##PLOT THE DATA
class Graph():

    def PlotData(self, epochs):

        index = 7

        # Extract the data of the selected epoch
        epoch_data = epochs[index].get_data()[0]
        times = epochs.times  # Get the time points

        # Compute the average across all channels
        avg_signal = np.mean(epoch_data, axis=0)

        plt.figure(figsize=(10, 5))
        plt.plot(times, avg_signal, label="Average Signal", color="blue")

        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (µV)")
        plt.title(f"Average Signal for Epoch {index}")
        plt.legend()
        plt.show()

##FEATURE EXTRACTION
class FeaturesExtraction:
    def Initialization(self):

        #Get the data from the epochs object
        epochsRight = mne.read_epochs('epochsRight-epo.fif') # shape is (n_epochs, n_channels, n_times)
        epochsLeft = mne.read_epochs('epochsLeft-epo.fif')  # shape is (n_epochs, n_channels, n_times)
        epochsIdle = mne.read_epochs('idle-epo.fif')  # shape is (n_epochs, n_channels, n_times)

        #Get the data from the epochs object
        epochsRight = epochsRight.get_data(copy=False)  # shape is (n_epochs, n_channels, n_times)
        #Get the data from the epochs object
        epochsLeft = epochsLeft.get_data(copy=False)  # shape is (n_epochs, n_channels, n_times)
        #Get the data from the epochs object
        epochsIdle = epochsIdle.get_data(copy=False)  # shape is (n_epochs, n_channels, n_times)

        #Extract features from all files
        featuresRight = self.ExtractFeatures(epochsRight)
        featuresLeft = self.ExtractFeatures(epochsLeft)
        featuresIdle = self.ExtractFeatures(epochsIdle)

        #Save the features to an excel file
        featuresRight.to_excel('featuresRight.xlsx', index=False)
        featuresLeft.to_excel('featuresLeft.xlsx', index=False)
        featuresIdle.to_excel('featuresIdle.xlsx', index=False)

    def ExtractFeatures(self, epochs):

        #Extract features
        TemporalFeatures = self.TemporalFeatures(epochs)
        SpectralFeatures = self.SpectralFeatures(epochs)

        mergedFeatures = pd.merge(TemporalFeatures, SpectralFeatures, on=['epoch', 'channel'], how='outer')

        mergedFeatures = mergedFeatures.drop(columns=['epoch', 'channel'])

        #Return the concatenated features
        return mergedFeatures

    def TemporalFeatures(self, epochs):

        # Point of maximum negativity
        def point_of_max_negativity(epochs):
            point_of_max_negativity= np.argmin(epochs, axis=-1)
            return point_of_max_negativity

        # Mean amplitude of the 4s data window
        def mean_amplitude(epochs):
            num_epochs = epochs.shape[0]
            num_channels = epochs.shape[1]
            averageList = np.zeros((num_epochs, num_channels))
            window05 = int(0.5*250)
            for epoch in range(epochs.shape[0]):

                for channel in range(epochs.shape[1]):
                    window_means=[]
                    for i in range(0, window05, windowSize):
                        window = epochs[epoch, channel, i:i + window05]
                        window_average = np.mean(window)
                        window_means.append(window_average)

                    averageList[epoch, channel] = np.mean(window_means)
            return averageList

        # Slope and intersection of a linear regression of the data in the 4s window
        def slope(epochs):
            # Initialize an array to store the slopes
            slopes = np.zeros((epochs.shape[0], epochs.shape[1]))

            # Loop through each epoch and each channel
            for epoch in range(epochs.shape[0]):
                for channel in range(epochs.shape[1]):
                    # Get the time points (assuming uniform sampling)
                    time_points = np.arange(epochs.shape[2])
                    # Fit a linear model (1st degree polynomial) to the data
                    slope, intercept = np.polyfit(time_points, epochs[epoch, channel, :], 1)
                    # Store the slope
                    slopes[epoch, channel] = slope

            return slopes

        def intercept(epochs):
            # Initialize an array to store the intercepts
            intercepts = np.zeros((epochs.shape[0], epochs.shape[1]))

            # Loop through each epoch and each channel
            for epoch in range(epochs.shape[0]):
                for channel in range(epochs.shape[1]):
                    # Get the time points (assuming uniform sampling)
                    time_points = np.arange(epochs.shape[2])
                    # Fit a linear model (1st degree polynomial) to the data
                    slope, intercept = np.polyfit(time_points, epochs[epoch, channel, :], 1)
                    # Store the intercept
                    intercepts[epoch, channel] = intercept

            return intercepts

        def variance(epochs):
            variances = compute_variance(epochs)
            return variances

        def meanNormal(epochs):
            mean = compute_mean(epochs)
            return mean

        def skewness(epochs):
            skewness = compute_skewness(epochs)
            return skewness

        def kurtosis(epochs):
            kurtosis = compute_kurtosis(epochs)
            return kurtosis

        def slopePriorDetection(data, detection_point, sfreq):
            # Calculate the number of samples corresponding to 0.5 seconds
            half_second_samples = int(0.5 * sfreq)
            # Extract the data window from 0.5 seconds prior to the detection point
            window_data = data[detection_point - half_second_samples:detection_point]
            # Get the time points for the window
            time_points = np.arange(len(window_data))
            # Fit a linear model to the data
            slope, _ = np.polyfit(time_points, window_data, 1)

            return slope

        def interceptPriorDetection(data, detection_point, sfreq):
            # Calculate the number of samples corresponding to 0.5 seconds
            half_second_samples = int(0.5 * sfreq)
            # Extract the data window from 0.5 seconds prior to the detection point
            window_data = data[detection_point - half_second_samples:detection_point]
            # Get the time points for the window
            time_points = np.arange(len(window_data))
            # Fit a linear model to the data
            _, intercept = np.polyfit(time_points, window_data, 1)
            return intercept

        def save_features_to_csv(features):
            feature_list = []
            for feature_name, feature_values in features.items():
                for epoch_idx in range(feature_values.shape[0]):
                    for channel_idx in range(feature_values.shape[1]):
                        feature_list.append({
                            'epoch': epoch_idx,
                            'channel': channel_idx,
                            feature_name: feature_values[epoch_idx, channel_idx]
                        })
            df = pd.DataFrame(feature_list)
            # Pivot the DataFrame to get features as columns
            df = df.pivot_table(index=['epoch', 'channel'],
                                      values=list(features.keys()),
                                      aggfunc='first').reset_index()
            return df

        features = {
            'point_of_max_negativity': point_of_max_negativity(epochs),
            'mean_amplitude': mean_amplitude(epochs),
            'linear_regression_slope': slope(epochs),
            'linear_regression_intercept': intercept(epochs),
            'variance': variance(epochs),
            'meanNormal': meanNormal(epochs),
            'skewness': skewness(epochs),
            'kurtosis': kurtosis(epochs),

        }

        # Save features to CSV file
        features = save_features_to_csv(features)
        return features

    def SpectralFeatures(self, epochs):

        def save_features_to_csv(features):
            feature_list = []
            for feature_name, feature_values in features.items():
                for epoch_idx in range(feature_values.shape[0]):
                    for channel_idx in range(feature_values.shape[1]):
                        feature_list.append({
                            'epoch': epoch_idx,
                            'channel': channel_idx,
                            feature_name: feature_values[epoch_idx, channel_idx]
                        })
            df = pd.DataFrame(feature_list)
            # Pivot the DataFrame to get features as columns
            df = df.pivot_table(index=['epoch', 'channel'],
                                      values=list(features.keys()),
                                      aggfunc='first').reset_index()

            return df

        def computePSD(epochs, frequencyBand):
            PSD = np.array([compute_pow_freq_bands(samplingFreq, epoch, freq_bands=frequencyBand) for epoch in epochs])
            return PSD

        #Define the frequency ranges for the different spectral bands: delta, theta, alpha, beta, and gamma.
        features={}

        for i in range(8,30):
            frequencyBand = [i, i+1]
            psd = computePSD(epochs, frequencyBand)
            features[f'PSD-{i}-{i+1}'] = psd

        # Save the features to a CSV file
        features=save_features_to_csv(features)

        return features

    def TimeScaleFeatures(self, epochs):

        def save_features_to_csv(features):
            feature_list = []

            for feature_name, feature_values in features.items():
                for epoch_idx in range(feature_values.shape[0]):
                    for channel_idx in range(feature_values.shape[1]):
                            feature_list.append({
                                'epoch': epoch_idx,
                                'channel': channel_idx,
                                feature_name: feature_values[epoch_idx, channel_idx]
                            })
            df = pd.DataFrame(feature_list)

            # Pivot the DataFrame to get features as columns
            df = df.pivot_table(index=['epoch', 'channel'],
                                      values=list(features.keys()),
                                      aggfunc='first').reset_index()

            return df


        # Compute sample entropy for each epoch
        wavelets= np.array([compute_wavelet_coef_energy(epoch, wavelet_name='db10') for epoch in epochs])

        features = {'wavelet': wavelets}

        # Save the features to a CSV file
        features = save_features_to_csv(features)

        return features

    def EntropyBasedFeatures(self, epochs):

        def save_features_to_csv(features):
            feature_list = []
            for feature_name, feature_values in features.items():
                for epoch_idx in range(feature_values.shape[0]):
                    for channel_idx in range(feature_values.shape[1]):
                        feature_list.append({
                            'epoch': epoch_idx,
                            'channel': channel_idx,
                            feature_name: feature_values[epoch_idx, channel_idx]
                        })
            df = pd.DataFrame(feature_list)
            # Pivot the DataFrame to get features as columns
            df = df.pivot_table(index=['epoch', 'channel'],
                                      values=list(features.keys()),
                                      aggfunc='first').reset_index()

            return df

        # Compute sample entropy for each epoch
        samp_entropy = np.array([compute_samp_entropy(epoch) for epoch in epochs])
        app_entropy = np.array([compute_app_entropy(epoch) for epoch in epochs])

        features = {'sample_entropy': samp_entropy,
                    'app_entropy': app_entropy}

        # Save the features to a CSV file
        features=save_features_to_csv(features)

        return features

def main():
    #Generate the instances and call the classes
    PrepareDataInstance = PrepareData()
    dataHandMovements70, dataHandMovements32, dataIdle = PrepareDataInstance.InitializeData()

    GenerateMNEArraysInstance = GenerateMNEArrays()
    infoMNE, ArrayHandMovements70, ArrayHandMovements32, ArrayIdle = GenerateMNEArraysInstance.GenerateRawArrayObject(dataHandMovements70, dataHandMovements32, dataIdle)

    FrequencySignalProcessingInstance = FrequencySignalProcessing()
    FilteredHandMovements70 = FrequencySignalProcessingInstance.SignalProcessing(ArrayHandMovements70, infoMNE)
    FilteredHandMovements32 = FrequencySignalProcessingInstance.SignalProcessing(ArrayHandMovements32, infoMNE)
    FilteredIdle = FrequencySignalProcessingInstance.SignalProcessing(ArrayIdle, infoMNE)

    DivideWindowsInstance = DivideWindows()
    EpochsRight, EpochsLeft, WindowsIdle, InfoMNE = DivideWindowsInstance.DivideWindows(infoMNE, FilteredHandMovements70, FilteredHandMovements32, FilteredIdle)

    GraphInstance = Graph()
    GraphInstance.PlotData(EpochsRight)

    SpatialSignalProcessingInstance = SpatialSignalProcessing()
    EpochsRight, EpochsLeft, WindowsIdle = SpatialSignalProcessingInstance.Initialization(EpochsRight, EpochsLeft, WindowsIdle, InfoMNE)

    GraphInstance.PlotData(EpochsRight)

    SaveFilesInstance = SaveFiles()
    SaveFilesInstance.SaveFiles(EpochsRight, EpochsLeft, WindowsIdle)

    FeaturesExtractionInstance = FeaturesExtraction()
    FeaturesExtractionInstance.Initialization()

if __name__ == '__main__':
    main()