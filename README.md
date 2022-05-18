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
First the dataset [charity_data.csv](https://github.com/bigoshunane/Deep-Learning-HM-16/blob/main/Resources/charity_data.csv) was processed by reading in data and noting the following target, feature, and identification variables:

-  Target Variable: `IS_SUCCESSFUL`
-  Feature Variables: `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, `ASK_AMT`
-  Identification Variables (to be removed): `EIN, NAME`

Then encoding categorical variables using `sklearn.preprocessing.OneHotEncoder` after bucketing noisy features `APPLICATION_TYPE` and `CLASSIFICATION` with many unique values. After one hot encoding, splitting data into the target and features, splitting again the data further into training and testing sets, and scale the training and testing data using `sklearn.preprocessing.StandardScaler`.

### 2. Compiling, Training, and Evaluating the Model
With preprocessed data, the base model defined in [AlphabetSoupCharity.ipynb](https://github.com/bigoshunane/Deep-Learning-HM-16/blob/main/AlphabetSoupCharity.ipynb) using `tensorflow.keras.models.Sequential` and `tensorflow.keras.layers.Dense` with the following parameters were built:

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
 


Optimization of the previous model by adjusting the parameters shown above and more in [AlphabetSoupCharity_Optimization.ipynb](https://github.com/bigoshunane/Deep-Learning-HM-16/blob/main/AlphabetSoupCharity_Optimization.ipynb), initially making the following single changes:

| Parameter | Value | Justification | Loss | Accuracy |
| --- |--- | --- | --- | -- |
| Training Duration (epochs) | Increase from 100 to 200 | Longer training time could result in more trends learned. | 0.575 | 0.730 |
| Hidden Layer Activation Function | Change from `relu` to `tanh` | Scaled data results in negative inputs which tanh does not output as zero. | 0.560 | 0.728 |
| Number of Input Features | Reduce from 43 to 34 by bucketing `INCOME_AMT` and `AFFILIATION` and dropping the redundant column `SPECIAL_CONSIDERATIONS_N` after encoding. | Less noise in the input data. | 0.560 | 0.733 |

A significant increase in performance from the initial model has not been seen; hence, did not meet 75% target accuracy criteria. To remedy this, a systematic approach by following Optimizing Neural Networks, by iteratively changing one model parameter at a time while holding others fixed, and then combining the parameters which generated the highest accuracy in each iterative search. This results in the following:

| Parameter | Value | Justification | Loss | Accuracy |
| --- |--- | --- | --- | -- |
| Training Duration (epochs) | [50, 100, 200, 300] | 200 | 0.571 | 0.732 |
| Architecture | All permutations with one to four hidden layers with 10, 30, 50, and 80 nodes, i.e [(10,), ..., (80,), (10, 30), (30, 10), ..., (80, 50), (10, 30, 50), (10, 50, 30), (30, 10, 50), ..., (80, 50, 30), (10, 30, 50, 80), (10, 30, 80, 50), (10, 50, 30, 80), ..., (80, 50, 30, 10)] | (80, 50, 30), i.e three hidden layers with 80, 50, and 30 nodes. | 0.570 | 0.730 |
| Hidden Layer Activation Function | [`relu`, `tanh`, `selu`, `elu`, `exponential`] | `tanh` | 0.551 | 0.734 |
| Number of Input Features | Bucket all combinations of `APPLICATION_TYPE`, `CLASSIFICATION`, `INCOME_AMT`, and `AFFILIATION`, similar structure as Architecture | Bucket `CLASSIFICATION` only (still drop redundant `SPECIAL_CONSIDERATIONS_N`) resulting in 50 input features. | 0.559 | 0.738 |
| Learning Rate | Coarse search [0.0001, 0.001, 0.01, 0.1, 1], fine search of six random values between 0.0001 and 0.01 | 0.0001144 | 0.557 | 0.731 |

Combining all optimized model parameters, we retrain and test to obtain the following testing loss and accuracy:

-  Loss: 0.554
-  Accuracy: 0.726

There is a negligible decrease in accuracy from the base model defined in [AlphabetSoupCharity.ipynb](https://github.com/bigoshunane/Deep-Learning-HM-16/blob/main/AlphabetSoupCharity.ipynb). As an additional optimization attempt, performing an iterative search of training batch size with values [1, 2, 4, 8, 16, 32, 64] and retaining the previously optimized parameters is possible. This search shows that a batch size of 64 yields the best results with a loss of 0.544 and accuracy of 0.736.

Considering the testing accuracy of each model, the architecture search generated the model with the best results and had the following parameters:

| Parameter | Value | 
| --- |--- | 
| Number of Hidden Layers | 3 |
| Architecture (hidden_nodes1, hidden_nodes2, hidden_nodes3) | (80, 50, 30) |
| Hidden Layer Activation Function | `relu` |
| Number of Output Nodes | 1 |
| Output Layer Activation Function | `sigmoid` |
| Learning Rate | 0.001 (default) |
| Training Duration (epochs) | 200 |
| Bucket Categorical Variables | No |
| Batch Size | 32 (default) |

Rebuilding and training this model, the summary shown in Optimized Model Summary and results shown in Optimized Model Training. While in this case the training accuracy reaches a promising 0.745, we find the model performance has decreased slightly when faced with the testing data:

-  Loss: 0.573
-  Accuracy: 0.729

## Summary

In summary, a deep neural network classification model that predicts loan applicant success from feature data contained in [charity_data.csv](https://github.com/bigoshunane/Deep-Learning-HM-16/blob/main/Resources/charity_data.csv) with 73% accuracy. This does not meet the 75% accuracy target, and the optimization methods employed here have not caused significant improvement.


## Additional Optimization Methods

Performance could increase through additional optimization techniques such as visualizing the numerical feature variable `ASK_AMT` to find and remove potential outliers that could be causing noise. Additionally, one could iteratively tune the parameters above and keep optimal values when moving to subsequent parameters instead of reverting to the base setting and combining after completion. This would however require more careful thought on the order with which one adjusts parameters to arrive at an optimized model.

##  Alternative Models
An alternative to the deep learning classification model presented in this project could be a more traditional Random Forest Classifier. This model is also appropriate for this binary classification problem and can often perform comparably to deep learning models with just two hidden layers. It is also advantageous in that there are less parameters to optimize and those which do require attention are more intuitive than those in a neural network.

## Usage
All code is contained in the Jupyter Notebook files `AlphabetSoupCharity.ipynb` and `AlphabetSoupCharity_Optimization.ipynb`. Therefore to replicate the results of this analysis, clone this repository and install the necessary dependencies into an isolated conda environment using the command:
`conda env create -f environment.yml`

On can then build, train, and test the classification model with baseline parameters by opening `AlphabetSoupCharity.ipynb` and running all cells. The user can then optimize this model by opening `AlphabetSoupCharity_Optimization.ipynb` and either running all cells (warning: the architecture and categorical bucketing iterative searches complete in roughly one hour), or by using the function build_train_test to perform additional iterative searches for optimal parameters with the following structure:

In [1]: learning_rates = [0.0001, 0.001, 0.01, 0.1, 1]                          

In [2]: results = []                                                            

In [3]: for rate in learning_rates: 
   ...:     result = build_train_test(learning_rate=rate, architecture=(80, 30), 
   
   ...:                               activation="relu", epochs=100, 
   
   ...:                               cat_cutoffs={"CLASSIFICATION": 1800}, 
   
   ...:                               batch_size=32) 
   
   ...:     results.append(result)
   
Here the default values were passed to parameters other than learning rate for clarity. The parameter architecture is a tuple whose length specifies the number of hidden layers and values the number of nodes in each layer. In this example there are two hidden layers, the first with 80 nodes and the second with 30. The parameter cat_cutoffs is a dictionary with keys specifying which categorical features should have bucketing and values the minimum number of unique occurences to stay out of the bucket. In this example, if a sample's value in `CLASSIFICATION` occurs less than 1800 times, its value is changed to OTHER. This function returns a tuple `(model_loss, model_accuracy)` which in this example is added to `results` for later analysis.

© 2021  Trilogy Education Services, a 2U, Inc. brand. All Rights Reserved.
