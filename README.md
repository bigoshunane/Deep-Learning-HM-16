# Neural Network: Charity Funding Predictor

## Background

The non-profit foundation Alphabet Soup wants to create an algorithm to predict whether or not applicants for funding will be successful. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

## Project Objective
Utlizing neural metwork and machine learnig to create a binary classifier capable of predicting whether the applicants for non-profit foundation Alphabet Soup will be sucessful or not for getting fund. 

## Resources

+  Data Source: charity_data.csv

+  Technologies used:

    -   Python 
    -   scikit-learn 
    -   pandas 
    -   TensorFlow 
    -   NumPy 
    -   Matplotlib 
    -   Jupyter Notebook 
    
## Overview of the analysis
This project includes Jupyter Notebook files that build, train, test, and optimize a deep neural network that models charity success from nine features in a loan application data set. TensorFlow Keras `Sequential` model with `Dense` hidden layers and a binary classification output layer and optimize this model by varying the following parameters are employed:

   + Training duration (in epochs)
   + Hidden layer activation functions
   + Hidden layer architecture
   + Number of input features through categorical variable binning
   + Learning rate
   + Batch size
   
## Results

###  1.  Data Preprocessing
First the dataset charity_data.csv was processed by reading in data and noting the following target, feature, and identification variables:

-  Target Variable: `IS_SUCCESSFUL`
-  Feature Variables: `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, `ASK_AMT`
-  Identification Variables (to be removed): `EIN, NAME`

Then encoding categorical variables using `sklearn.preprocessing.OneHotEncoder` after bucketing noisy features `APPLICATION_TYPE` and `CLASSIFICATION` with many unique values. After one hot encoding, splitting data into the target and features, splitting again the data further into training and testing sets, and scale the training and testing data using `sklearn.preprocessing.StandardScaler`.

### 2. Compiling, Training, and Evaluating the Model
With preprocessed data, the base model defined in AlphabetSoupCharity.ipynb using `tensorflow.keras.models.Sequential` and `tensorflow.keras.layers.Dense` with the following parameters were built:

| Parameter | Value | Justification |
| --- |--- | --- |
| Number of Hidden Layers | 2 | Deep neural network is necessary for complex data, good starting point with low computation time. |
| Architecture (hidden_nodes1, hidden_nodes2) | (80,30) | First layer has roughly two times the number of inputs (43), smaller second layer offers shorter computation time. |
| Hidden Layer Activation Function | `relu` | Simple choice for inexpensive training with generally good performance. |
| Number of Output Nodes | 1 | Model is a binary classifier and should therefore have one output predicting if `IS_SUCCESSFUL` is `True` or `False`. |
| Output Layer Activation Function | `sigmoid` | Provides a probability output (value between 0 and 1) for the classification of `IS_SUCCESSFUL`. |

This yields the model summary shown in Base Model Summary. Then comping and training the model using the `binary_crossentropy` loss function, `adam` optimizer, and `accuracy` metric to obtain the training results shown in Base Model Training. Verifying with the testing set, the following results obtained:

-  Loss: 0.561
-  Accuracy: 0.729
 


