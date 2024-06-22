# Mini-Project for Fundamentals of Machine Learning Course
![background](https://media.licdn.com/dms/image/C5612AQFyWMfAbjteMw/article-cover_image-shrink_600_2000/0/1520134891908?e=2147483647&v=beta&t=zr9SBsdj7BZtz-phdEyuyRoX4fKlk04TOZ_D_o9atwQ)

## 📑 Project Policy
- Team: 

    |No.| Student Name    | Student ID |
    | --------| -------- | ------- |
    |1|Trịnh Minh Anh| 21280005|
    |2|Nguyễn Phúc Gia Nghi| 21280035|
    |3|Nguyễn Lưu Phương Ngọc Lam| 21280096|

## 📌 Problem Introduction
Utilizing Machine Learning algorithms for the topic of facial expression recognition classification in our third-year final mini-project, we aim to systematically present our methodology, thoroughly analyze the results, and draw comprehensive conclusions. 

**Problem:** Given a set of grayscale images of faces represented as pixel arrays, our task is to build a model that detects facial expressions from the input images. This type of problem is known as the *Facial Expression Recognition Challenge (FER2013)*
- The input: A set of images depicting emotions, with each emotion assigned a numerical label in one of seven categories.
- The output: The trained models capable of predicting the expression from new facial image data.


## 🧩 Data Exploring
Regarding the EDA section, we summarized the information as follows:
- Our dataset comprises 7 labels with specific encodings: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral.
- The dataset is free of NaN values.
- There is one image contains one unique value, meaning this image consists entirely of one color.
- Some duplicate images have been found and removed.
- There exists an imbalance in the distribution of labels, notably with label 1 appearing significantly less frequently compared to the other labels

    

## 🔍 Methodology Overview
In this study, we will use 4 models: XGBoost, CatBoost, LightGBM, and MLP. The model application will be based on 3 types of input data: the original data (before PCA), the data after applying PCA, and the data processed using an AutoEncoder. 

### XGBoost (eXtreme Gradient Boosting)
XGBoost is a powerful ensemble learning algorithm that enhances prediction accuracy through gradient boosting. The core idea is to iteratively add decision trees to minimize the residual errors of previous models.

**Model Training:**

- Handle imbalanced classes by computing class weights and scaling the positive class weight accordingly.
- Define the XGBoost model with parameters like maximum depth, tree method, and device setting.
- Use a grid search to optimize hyperparameters such as the number of `estimators`, `gamma`, `regularization terms`, and `learning rate`.


### CatBoost (Categorical Boosting)
CatBoost is tailored to efficiently handle categorical features, which are common in FER datasets. It uses ordered boosting and permutation techniques to manage categorical data during training.
Model Training:

- Compute class weights to handle imbalanced classes.
- Define the CatBoost model with parameters like depth and task type (GPU).
- Use a grid search to optimize hyperparameters such as `learning rate`, `iterations`, and `L2 regularization`.
### LightGBM (Light Gradient Boosting Machine)
LightGBM focuses on training efficiency and scalability. It employs techniques like Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) to enhance performance.

**Model Training:**

- Compute class weights to handle imbalanced classes and create a class weight dictionary.
- Define the LightGBM model with parameters such as objective, number of classes, and device setting (GPU).
- Create a pipeline combining feature scaling and the LightGBM classifier.
- Use a grid search to optimize hyperparameters like the number of `leaves`, `learning rate`, number of `estimators`, `max depth`, and `regularization terms`.

### MLP (Multi-Layer Perceptron)
An MLP belongs to a class of feedforward artificial neural networks. It has the remarkable capability of capturing intricate patterns in data through the use of multiple layers of neurons, each employing non-linear activation functions. 

**Model Architecture:** The model consists of three fully connected (dense) layers:
- The input layer is responsible for receiving the initial flattened input features. 
- The hidden layers consist of two layers, one with **128 neurons** and another with **32 neurons**. Each hidden layer is followed by an `Exponential Linear Unit` (ELU) activation function and batch normalization, which enhance learning stability and convergence. 
- The output layer generates a vector of size 7, corresponding the different facial expression classes. It applies a `softmax` activation function to produce probability distributions across these classes.

**Model Training:**

- The MLP is trained using backpropagation with a suitable loss function, such as cross-entropy loss, to minimize the error between predicted and actual class labels.
- Optimization algorithms like Adam are used to update the weights and biases of the network during training.


### Brief to the Augmentation Data


## 📦 Project Structure

The repository is organized into the following directories:

- **/data**: This directory contains the facial expression dataset. You'll need to download the dataset and place it here before running the notebooks. (Download link provided below)
- **/notebooks**: This directory contains the Jupyter notebook ```EDA.ipynb```. This notebook guides you through exploratory data analysis (EDA) and classification tasks.



## 🧱 Our proposed methods

### Dealing with unbalanced data

At first, we "unzip" data into folders for futher process

```
|- data
    |- images
        |- train
            |- 0
            ...
            |- 6
        |- Valid
            |- 0
            ...
            |- 6
        |- test
            |- 0
            ...
            |- 6
```

<p align="center">
    <img src="https://cdn.discordapp.com/attachments/893738488137142286/1254061057375338506/counting_labels.png?ex=66781edf&is=6676cd5f&hm=c2b7f5a55817a5ca0eb11490fa33eca92f4b97756fb664ffe4431d0ed5c67241&" width="500" align="center">
</p>

Looking at counting plot above, we can see that our data is heavy unbalaced, so we devided it into train/valid/test subset and then choosed to downsample data having label 3 to 4000 samples and upsample data in training set with label 0, 1, 2, 4, 5, 6 to 4000 using some image augmentation methods: `RandomRotation`, `RandomHorizontalFlip`, `ColorJitter`, `RandomAdjustSharpness` and `Normalize`.

### Image encoding

To encode image, we simply use a simple CNN-based Auto encoder

![](https://cdn.discordapp.com/attachments/893738488137142286/1254078699461152768/autoencoder.png?ex=66782f4d&is=6676ddcd&hm=595078436c281af940feeadadd5922aabc6d71741e4e9b26de1a73a760fe820e&)

It can be seen that our auto encoder model can extract features good!

### Facial classification

Then we use pretrained-encoder to vectorize input images into 256 dimensions vectors and pass them into machine learning models to predict facial expression.

![](https://cdn.discordapp.com/attachments/893738488137142286/1254078699028877342/meomeo.png?ex=66782f4d&is=6676ddcd&hm=8a92efc5a99ed0062d7126884a545665d23622cf1392a4d2c283965b6238e1ec&)

## 🏁 Experimental Results

|   | Features Extractor | Model | Accuracy | Precision | Recall | F1-score |
|:-:|:-|:-:|:-|:-|:-|:-|
| **0** | **None** | **XGBoost** | **0.4419** | **0.44** | **0.44** | **0.42** |
| 1 | None | CatBoost | 0.3254 | 0.33 | 0.33 | 0.31 |
| 2 | None | LightGBM | 0.4282 | 0.43 | 0.43 | 0.42 |
| 3 | None | MLP | 0.3628 | 0.31 | 0.36 | 0.33 |
| 4 | PCA  | XGBoost | 0.3266 | 0.32 | 0.33 | 0.31 |
| 5 | PCA  | CatBoost | 0.1780 | 0.03 | 0.18 | 0.05 |
| 6 | PCA  | LightGBM | 0.4272 | 0.06 | 0.25 | 0.10 |
| 7 | PCA  | MLP | 0.4016 | 0.40 | 0.40 | 0.39 |
| 8 | Auto encoder | CatBoost | 0.38 | 0.44 | 0.38 | 0.39 |
| 9 | Auto encoder | XGBoost | 0.41 | 0.41 | 0.41 | 0.41 |
| 10 | Auto encoder | LightGBM | 0.39 | 0.39 | 0.39 | 0.39 |

The above result table indicates that:

| Models | Descriptions |
|:-|:-|
|XGBoost with the original data (None) has the highest accuracy among the three input data, with 0.4419. |This can be attributed to the original data providing richer information about critical features. Besides, XGBoost, being a powerful ensemble algorithm, excels in learning complex, diverse and informative patterns from data.|
|LightGBM with data after PCA ranks second with an accuracy of 0.4272 | This suggests that PCA may not effectively eliminate non-essential information for gradient-boosting algorithms like LightGBM|
|XBoost with data using Autoencoder has an accuracy of 0.41, the lowest among the three types of input data, but still the highest within the Auto Encoder group. |Auto Encoder was used to learn latent features of the data; however, in this case, the re-encoding process may not have provided sufficient information for XGBoost to perform as effectively as with the Original data. |

