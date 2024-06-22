# Mini-Project for Fundamentals of Machine Learning Course
![background](https://media.licdn.com/dms/image/C5612AQFyWMfAbjteMw/article-cover_image-shrink_600_2000/0/1520134891908?e=2147483647&v=beta&t=zr9SBsdj7BZtz-phdEyuyRoX4fKlk04TOZ_D_o9atwQ)

## 📌 Problem Introduction
Utilizing Machine Learning algorithms for the topic of facial expression recognition classification in our third-year final mini-project, we aim to systematically present our methodology, thoroughly analyze the results, and draw comprehensive conclusions. 

**Problem:** Given a set of grayscale images of faces represented as pixel arrays, our task is to build a model that detects facial expressions from the input images. This type of problem is known as the *Facial Expression Recognition Challenge (FER2013)*
- The input: A set of images depicting emotions, with each emotion assigned a numerical label in one of seven categories.
- The output: The trained models capable of predicting the expression from new facial image data.


## 📑 Project Policy
- Team: 

    |No.| Student Name    | Student ID |
    | --------| -------- | ------- |
    |1|Trịnh Minh Anh| 21280005|
    |2|Nguyễn Phúc Gia Nghi| 21280035|
    |3|Nguyễn Lưu Phương Ngọc Lam| 21280096|

## 🔍 Methodology Overview
In this study, we will use 4 models: XGBoost, CatBoost, LightGBM, and MLP. The model application will be based on 3 types of input data: the original data (before PCA), the data after applying PCA, and the data processed using an AutoEncoder.

### XGBoost (eXtreme Gradient Boosting)
XGBoost is a powerful ensemble learning algorithm that enhances prediction accuracy through gradient boosting. The core idea is to iteratively add decision trees to minimize the residual errors of previous models.


## 📦 Project Structure

The repository is organized into the following directories:

- **/data**: This directory contains the facial expression dataset. You'll need to download the dataset and place it here before running the notebooks. (Download link provided below)
- **/notebooks**: This directory contains the Jupyter notebook ```EDA.ipynb```. This notebook guides you through exploratory data analysis (EDA) and classification tasks.

## ⚙️ Usage

This project is designed to be completed in the following steps:

1. **Fork the Project**: Click on the ```Fork``` button on the top right corner of this repository, this will create a copy of the repository in your own GitHub account. Complete the table at the top by entering your team member names.

2. **Download the Dataset**: Download the facial expression dataset from the following [link](https://mega.nz/file/foM2wDaa#GPGyspdUB2WV-fATL-ZvYj3i4FqgbVKyct413gxg3rE) and place it in the **/data** directory:

3. **Complete the Tasks**: Open the ```notebooks/EDA.ipynb``` notebook in your Jupyter Notebook environment. The notebook is designed to guide you through various tasks, including:
    
    1. Prerequisite
    2. Principle Component Analysis
    3. Image Classification
    4. Evaluating Classification Performance 

    Make sure to run all the code cells in the ```EDA.ipynb``` notebook and ensure they produce output before committing and pushing your changes.

5. **Commit and Push Your Changes**: Once you've completed the tasks outlined in the notebook, commit your changes to your local repository and push them to your forked repository on GitHub.


Feel free to modify and extend the notebook to explore further aspects of the data and experiment with different algorithms. Good luck.

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

Looking at counting plot above, we can see that our data is heavy unbalace, so we devided it into train/valid/test subset and then choosed to downsample data having label 3 to 4000 samples and upsample data in training set with label 0, 1, 2, 4, 5, 6 to 4000 using some image augmentation methods: `RandomRotation`, `RandomHorizontalFlip`, `ColorJitter`, `RandomAdjustSharpness` and `Normalize`.

### Image encoding

To encode image, we simply use a simple CNN-based Auto encoder

![](https://cdn.discordapp.com/attachments/893738488137142286/1254059327015358594/autoencoder.png?ex=66781d42&is=6676cbc2&hm=4b868338a191b5bcfe332d98d402ec4eebec6c4bc835e2321c5dfeb616c5edf6&)

It can be seen that our auto encoder model can extract features good!

### Facial classification

Then we use pretrained-encoder to vectorize input images into 256 dimensions vectors and pass them into machine learning models to predict facial expression.

![](https://cdn.discordapp.com/attachments/893738488137142286/1254059961416421517/meomeo.png?ex=66781dda&is=6676cc5a&hm=e2d2502058b3bff6ad1df19f261bb5d951891268abed0ab86a5c8e462d34b2b7&)

## 🏁 Experimental Results

|   | Features Extractor | Model | Accuracy |
|:-:|:-|:-:|:-|
| **0** | **None** | **XGBoost** | **0.4419** |
| 1 | None | CatBoost | 0.3254 |
| 2 | None | LightGBM | 0.4282 |
| 3 | None | MLP | 0.3628 |
| 4 | PCA  | XGBoost | 0.3266 |
| 5 | PCA  | CatBoost | 0.3254 |
| 6 | PCA  | LightGBM | 0.4272 |
| 7 | PCA  | MLP | 0.4016 |
| 8 | Auto encoder | CatBoost | 0.38 |
| 9 | Auto encoder | XGBoost | 0.41 |
| 10 | Auto encoder | LightGBM | 0.39 |

