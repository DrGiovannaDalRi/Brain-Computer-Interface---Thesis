###MACHINE LEARNING CLASSIFICATION OF EEG DATA WITH PERFORMANCE ANALYSIS

"""
Pipeline:

1. Load augmented EEG feature datasets
2. Build two binary classification problems:

   Classifier 1:
   - Movement vs Idle

   Classifier 2:
   - Right-hand vs Left-hand motor imagery

3. Generate class labels
4. Split data into training and testing sets
    - Augmented features are used only as training, not as testing
5. Train Random Forest classifiers
6. Perform hyperparameter optimization using GridSearchCV
7. Evaluate classifier performance using:
   - Accuracy
   - Confusion Matrix
   - Precision
   - Recall
   - F1-score
   - Cross-validation

8. Compute SHAP explainability analysis
9. Identify the most informative EEG features
10. Export evaluation reports
11. Save trained models

Output:
- EEGModelMovementVSIdle.pkl
- EEGModelLeftVSRight.pkl
- Evaluation reports (.txt)
- SHAP importance tables (.xlsx)
"""


##IMPORT LIBRARIES

#Libraries needed for the handling and plotting of the data
import numpy as np
import pandas as pd

#Libraries needed for the feature classification
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

#Libraries for dimensionality reduction PCA
from sklearn.decomposition import PCA

#Libraries for feature selection
from sklearn.ensemble import RandomForestClassifier

#Libraries for dimensionality reduction
from sklearn.preprocessing import StandardScaler

#Libraries for explainability and fine-tuning of features
import shap

#Libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns

##DEFINITION OF THE NAMES OF USED FILES
MOVEMENTVSIDLE='EEGModelMovementVSIdle.pkl'
LEFTVSRIGHT='EEGModelLeftVSRight.pkl'

TXTMOVEMENTVSIDLE='EvaluationMovementVSIdle.txt'
TXTLEFTVSRIGHT='EvaluationLeftVSRight.txt'

SHAPMOVEMENTVSIDLE='SHAPMovementVSIdle.xlsx'
SHAPLEFTVSRIGHT='SHAPLeftVSRight.xlsx'

FEATURESRIGHT_REAL = 'featuresRight.xlsx'
FEATURESLEFT_REAL = 'featuresLeft.xlsx'
FEATURESIDLE_REAL = 'featuresIdle.xlsx'

FEATURESRIGHT_AUG = 'AugmentationFeaturesRight.xlsx'
FEATURESLEFT_AUG = 'AugmentationFeaturesLeft.xlsx'
FEATURESIDLE_AUG = 'AugmentationFeaturesIdle.xlsx'

##DEFINITION OF THE VARIABLES
MOVEMENTLABEL = 1
IDLELABEL = 0
RIGHTLABEL = 1
LEFTLABEL = 0

class FeaturesClassification:
    #IMPORT EXTRACTED FEATURES
    def Initialization(self):

        movementVSidleFeaturesReal, movementVSidleFeaturesAugmented = self.movementVSidleFeatures()
        rightVSleftFeaturesReal, rightVSleftFeaturesAugmented = self.rightVSleftFeatures()

        return movementVSidleFeaturesReal, movementVSidleFeaturesAugmented, rightVSleftFeaturesReal, rightVSleftFeaturesAugmented

    def movementVSidleFeatures(self):
        # Load real features
        rightFeaturesReal = pd.read_excel(FEATURESRIGHT_REAL)
        leftFeaturesReal = pd.read_excel(FEATURESLEFT_REAL)
        idleFeaturesReal = pd.read_excel(FEATURESIDLE_REAL)

        # Load augmented features
        rightFeaturesAugmented = pd.read_excel(FEATURESRIGHT_AUG)
        leftFeaturesAugmented = pd.read_excel(FEATURESLEFT_AUG)
        idleFeaturesAugmented = pd.read_excel(FEATURESIDLE_AUG)

        # Generate labels, taking into consideration the 2-class + 2-class problem
        rightFeaturesReal['Label'] = MOVEMENTLABEL #1 class= movement
        leftFeaturesReal['Label'] = MOVEMENTLABEL #1 class= movement
        idleFeaturesReal['Label'] = IDLELABEL #0 class = idle/no movement

        rightFeaturesAugmented['Label'] = MOVEMENTLABEL
        leftFeaturesAugmented['Label'] = MOVEMENTLABEL
        idleFeaturesAugmented['Label'] = IDLELABEL

        # Combine data in a dataframe allFeatures
        allFeaturesReal = pd.concat([rightFeaturesReal, leftFeaturesReal, idleFeaturesReal], axis=0)
        allFeaturesAugmented = pd.concat([rightFeaturesAugmented, leftFeaturesAugmented, idleFeaturesAugmented], axis=0)

        return allFeaturesReal, allFeaturesAugmented

    def rightVSleftFeatures(self):
        # Load real features
        rightFeaturesReal = pd.read_excel(FEATURESRIGHT_REAL)
        leftFeaturesReal = pd.read_excel(FEATURESLEFT_REAL)

        # Load augmented features
        rightFeaturesAugmented = pd.read_excel(FEATURESRIGHT_AUG)
        leftFeaturesAugmented = pd.read_excel(FEATURESLEFT_AUG)

        # Generate labels, taking into consideration the 2-class + 2-class problem
        rightFeaturesReal['Label'] = RIGHTLABEL #1 class= right movement
        leftFeaturesReal['Label'] = LEFTLABEL #0 class= left movement

        rightFeaturesAugmented['Label'] = RIGHTLABEL #1 class= right movement
        leftFeaturesAugmented['Label'] = LEFTLABEL #0 class= left movement

        # Combine data in a dataframe allFeatures
        allFeaturesReal = pd.concat([rightFeaturesReal, leftFeaturesReal], axis=0)

        allFeaturesAugmented = pd.concat([rightFeaturesAugmented, leftFeaturesAugmented], axis=0)

        return allFeaturesReal, allFeaturesAugmented
class ModelDefinition():

    #Data splitting
    def DataSplitting(self, allFeaturesReal, allFeaturesAugmented, NAMEMODEL, NAMETXT, NAMESHAP):
        self.NAMEMODEL = NAMEMODEL
        self.NAMETXT = NAMETXT
        self.NAMESHAP = NAMESHAP
        self.allFeaturesReal = allFeaturesReal
        self.allFeaturesAugmented = allFeaturesAugmented

        # Separate features and labels

        #Real features
        #X= Features - input data
        self.X_Real = self.allFeaturesReal.drop('Label', axis=1)
        #y= Labels - target data
        self.y_Real = self.allFeaturesReal['Label']

        #Augmented features
        #X= Features - input data
        self.X_Aug = self.allFeaturesAugmented.drop('Label', axis=1)
        #y= Labels - target data
        self.y_Aug = self.allFeaturesAugmented['Label']

        # Split data into training and testing sets
        #Training set is composed of both augmented and real features
        #Test set is composed of only real features
        #X_train = Training data - input data
        #X_test = Testing data - input data
        #y_train = Training data - target data, corresponding to X_train
        #y_test = Testing data - target data, corresponding to X_test
        #test_size = 20% of the data will be used for testing, 80% for training
        #random_state = 42 (seed value), so that the data is split in the same way every time the code runs
        self.X_train_real, self.X_test, self.y_train_real, self.y_test = train_test_split(self.X_Real, self.y_Real, test_size=0.2, random_state=42)

        self.X_train = pd.concat([self.X_train_real, self.X_Aug], axis=0, ignore_index=True)

        self.y_train = pd.concat([self.y_train_real, self.y_Aug], axis=0, ignore_index=True)


        #Run the features standardization code
        #Even if the features augmentation had a standardization, they are then saved in the original scale after inverse_transform()
        # --> Random Forest does not require feature scaling, so it is commented with this specific ML model

        #self.StandardizeFeatures()

    #FEATURES SCALING
    def StandardizeFeatures(self):
        # Standardize the features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    #MODEL TRAINING
    #Classifiers' codes are currently separated in the eventuality the parameters differ
    def ModelTrainingFirstClassifier(self):
        # Define the Random Forest model
        modelRF = RandomForestClassifier(bootstrap=False, criterion='entropy', max_depth=20, max_features='log2', min_samples_leaf=1, min_samples_split=2)

        # Hyperparameter tuning using Grid Search
        param_grid = {'n_estimators': [128, 200, 500]}

        # Uses cross-validation to find the best hyperparameters
        # The cv=5 parameter specifies 5-fold cross-validation
        self.grid = GridSearchCV(modelRF, param_grid, refit=True, cv=5, verbose=2)
        self.grid.fit(self.X_train, self.y_train)

        # Best parameters and predictions
        # Obtain the best hyperparameters from the grid search
        self.bestParameters = self.grid.best_params_
        self.bestModel = self.grid.best_estimator_

        # Use the best model found by the grid search to make predictions on the test data
        # Uses the best model to predict the labels for the test data.
        self.predictions = self.grid.predict(self.X_test)

        #Run model evaluation function
        self.ModelEvaluation()

    #MODEL TRAINING
    def ModelTrainingSecondClassifier(self):
        # Define the Random Forest model
        modelRF = RandomForestClassifier(bootstrap=False, criterion='entropy', max_depth=20, max_features='log2', min_samples_leaf=1, min_samples_split=2)

        # Hyperparameter tuning using Grid Search
        param_grid = {'n_estimators': [128, 200, 500]}

        # Uses cross-validation to find the best hyperparameters
        # The cv=5 parameter specifies 5-fold cross-validation
        self.grid = GridSearchCV(modelRF, param_grid, refit=True, cv=5, verbose=2)
        self.grid.fit(self.X_train, self.y_train)

        # Best parameters and predictions
        # Obtain the best hyperparameters from the grid search
        self.bestParameters = self.grid.best_params_
        self.bestModel = self.grid.best_estimator_

        # Use the best model found by the grid search to make predictions on the test data
        # Uses the best model to predict the labels for the test data.
        self.predictions = self.grid.predict(self.X_test)

        #Run model evaluation function
        self.ModelEvaluation()

    #MODEL EVALUATION
    def ModelEvaluation(self):
        #Gives evaluation metrics
        #Evaluate the model using classification metrics
        #Includes: precision, recall, f1-score, support
        classificationReport = classification_report(self.y_test, self.predictions)
        print("The classification report is:")
        print(classificationReport)

        #Evaluate the model using confusion matrix
        #Includes: true positives, true negatives, false positives, false negatives
        confusionMatrix=confusion_matrix(self.y_test, self.predictions)
        print("The confusion matrix is:")
        print(confusionMatrix)

        # Plotting the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusionMatrix, annot=True, fmt='d', cmap='Greens', xticklabels=["Class 1", "Class 2"],
                    yticklabels=["Class 1", "Class 2"])
        plt.title("Confusion Matrix")
        plt.ylabel("True Labels")
        plt.xlabel("Predicted Labels")
        plt.show()

        #Evaluate the model using accuracy score
        accuracyScore=accuracy_score(self.y_test, self.predictions)
        print("The accuracy score is:")
        print(accuracyScore)

        # Cross-validation scores
        #This line performs cross-validation on the entire dataset (X, y) using the best estimator found by the grid search
        #The cv=5 parameter specifies 5-fold cross-validation
        #Meaning the data is split into 5 subsets, and the model is trained and evaluated 5 times, each time using a different subset as the validation set and the remaining subsets as the training set
        #The result is an array of scores, one for each fold.
        cv_scores = cross_val_score(self.grid.best_estimator_, self.X_train, self.y_train, cv=5)

        #The scores give an idea of how the model performs on different subsets of the data
        print("Cross-validation scores:", cv_scores)
        mean_cv_score = np.mean(cv_scores)

        #The mean of the cross-validation scores provides a single metric that summarizes the model’s performance across all folds
        print("Mean cross-validation score:", np.mean(cv_scores))

        # Save the evaluation metrics to a file
        with open(self.NAMETXT, 'w') as f:
            f.write("The classification report is:\n")
            f.write(classificationReport + "\n")
            f.write("The confusion matrix is:\n")
            f.write(np.array2string(confusionMatrix) + "\n")
            f.write("The accuracy score is:\n")
            f.write(str(accuracyScore) + "\n")
            f.write("Cross-validation scores:\n")
            f.write(np.array2string(cv_scores) + "\n")
            f.write("Mean cross-validation score:\n")
            f.write(str(mean_cv_score) + "\n")
            f.write("Used parameters:\n")
            f.write(str(self.bestParameters) + "\n")

        print(f"Evaluation saved as a txt file: {self.NAMETXT}")

        #Compute the SHAP
        self.SHAPExplainability()

        #Save the model
        self.SaveBestModel()

    def SHAPExplainability(self):

        # Initialize SHAP explainer
        explainer = shap.Explainer(self.bestModel, self.X_train)

        # Compute SHAP values for the test dataset
        shap_values = explainer(self.X_test)
        plt.yticks(fontsize=10)

        print("SHAP beeswarm for label = 0")
        plt.subplots_adjust(left=0.30)
        shap.plots.beeswarm(shap_values[..., 0], max_display=15)
        plt.gcf().set_size_inches(16, 8)
        plt.tight_layout()

        print("SHAP beeswarm for label = 1")
        plt.subplots_adjust(left=0.30)
        shap.plots.beeswarm(shap_values[..., 1], max_display=15)
        plt.gcf().set_size_inches(16, 8)
        plt.tight_layout()

        feature_importance_dim1 = np.abs(shap_values.values[..., 0]).mean(axis=0)
        feature_importance_dim2 = np.abs(shap_values.values[..., 1]).mean(axis=0)

        best_feature_dim1 = self.X_test.columns[np.argmax(feature_importance_dim1)]
        worst_feature_dim1 = self.X_test.columns[np.argmin(feature_importance_dim1)]

        best_feature_dim2 = self.X_test.columns[np.argmax(feature_importance_dim2)]
        worst_feature_dim2 = self.X_test.columns[np.argmin(feature_importance_dim2)]

        print(f"Dimension 1 - Most important feature: {best_feature_dim1}, Least important feature: {worst_feature_dim1}")
        print(f"Dimension 2 - Most important feature: {best_feature_dim2}, Least important feature: {worst_feature_dim2}")

        # Save SHAP values to an Excel file (transposed)
        feature_names = self.X_test.columns
        df_shap = pd.DataFrame({
            "SHAP Value (Dim 1)": feature_importance_dim1,
            "SHAP Value (Dim 2)": feature_importance_dim2
        }, index=feature_names)

        df_shap.to_excel(self.NAMESHAP)
        
    def SaveBestModel(self):
        # Save the best model found by GridSearchCV
        joblib.dump(self.bestModel, self.NAMEMODEL)
        print(f"Best model saved as: {self.NAMEMODEL}")

def main():
    #Definition of the instance of the class FeatureClassification
    FeaturesClassificationInstance = FeaturesClassification()
    #Call of the function Initialization in the FeaturesClassification class
    movementVSidleFeaturesReal, movementVSidleFeatureAugmented, rightVSleftFeaturesReal, rightVSleftFeaturesAugmented = FeaturesClassificationInstance.Initialization()

    #Definition of the instance of the class ModelDefinition
    ModelDefinitionInstance = ModelDefinition()
    #Call of the function DataSplitting in the ModelDefinition class for the first model (Movement VS Idle)
    ModelDefinitionInstance.DataSplitting(movementVSidleFeaturesReal, movementVSidleFeatureAugmented, MOVEMENTVSIDLE, TXTMOVEMENTVSIDLE, SHAPMOVEMENTVSIDLE)
    #Call of the function DataSplitting in the ModelDefinition class for the first model (Movement VS Idle)
    ModelDefinitionInstance.ModelTrainingFirstClassifier()

    #Call of the function DataSplitting in the ModelDefinition class for the second model (Left VS Right)
    ModelDefinitionInstance.DataSplitting(rightVSleftFeaturesReal, rightVSleftFeaturesAugmented, LEFTVSRIGHT, TXTLEFTVSRIGHT, SHAPLEFTVSRIGHT)
    #Call of the function DataSplitting in the ModelDefinition class for the first model (Movement VS Idle)
    ModelDefinitionInstance.ModelTrainingSecondClassifier()

if __name__ == '__main__':
    main()
