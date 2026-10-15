# Iris Flower Classification Using Machine Learning

## Overview

How accurately can a machine learning model identify a flower species from its physical measurements?

This project explores that question using the Iris dataset, a classic classification dataset in machine learning. It implements two supervised learning algorithms — **Decision Tree and K-Nearest Neighbors (KNN)** — to predict the species of an Iris flower from its sepal and petal measurements.

The objective is to understand the complete classification workflow, from data preparation and model training to performance evaluation and comparison.

## Dataset

The Iris dataset contains 150 samples across three flower species:

- Iris setosa
- Iris versicolor
- Iris virginica

Each sample contains four numerical features:

| Feature | Description |
|---|---|
| Sepal length | Length of the sepal |
| Sepal width | Width of the sepal |
| Petal length | Length of the petal |
| Petal width | Width of the petal |

**Target variable:** Iris species

**Dataset source:** [UCI Machine Learning Repository — Iris Dataset](https://archive.ics.uci.edu/dataset/53/iris)

The dataset is loaded directly using `sklearn.datasets.load_iris()`, so downloading a separate dataset file is not required.

## Tools and Libraries

- Python 3
- Jupyter Notebook / Google Colab
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn

## Workflow

### 1. Data Preparation

- Loaded and inspected the dataset.
- Checked for missing values.
- Separated the four flower measurements into features (`X`) and species labels into the target (`y`).
- Split the dataset into training and testing sets using an 80:20 ratio with stratification to preserve class proportions.

### 2. Model Building

**Decision Tree Classifier**

A tree-based model that learns decision rules from the training data to classify flowers based on their measurements.

**K-Nearest Neighbors (KNN)**

A distance-based model that predicts a flower's species using the labels of its nearest training samples. This project uses `k = 5`.

### 3. Evaluation

Both models were evaluated using test-set accuracy and confusion matrices. Accuracy measures the proportion of test samples classified correctly, while confusion matrices show the distribution of correct and incorrect predictions across the three species.

## Results

| Model | Test Accuracy |
|---|---:|
| Decision Tree | 93.33% |
| K-Nearest Neighbors (KNN) | 100% |

On the selected test split, KNN achieved higher accuracy than the Decision Tree model.

The confusion matrices in the notebook provide a more detailed view of each model's classification performance.

*Note: These results reflect one train-test split. They do not guarantee the same accuracy on different splits or unseen datasets.*

## Key Learnings

- Preparing features and target labels for supervised learning.
- Training and comparing two classification algorithms.
- Evaluating predictions using accuracy and confusion matrices.
- Understanding that model performance depends on the dataset, training process, and evaluation split.

## How to Run

### Requirements

Python 3 and the libraries listed below.

Install the dependencies:

```bash
pip install pandas scikit-learn matplotlib seaborn jupyter
```

### Run the Notebook

1. Clone or download this repository.
2. Open `iris_classification.ipynb` in Jupyter Notebook, or upload it to Google Colab.
3. Run the cells in order to reproduce the data preparation, model training, and evaluation.

## Project Structure

```text
iris-classification/
├── iris_classification.ipynb
└── README.md
```

## Conclusion

This project demonstrates a basic end-to-end machine learning classification workflow. Both Decision Tree and KNN were trained on the Iris dataset and evaluated using test accuracy and confusion matrices. KNN achieved the higher accuracy on the selected test split, while the comparison provided practical insight into how different classification algorithms approach the same prediction task.

The project serves as a foundation for exploring more advanced topics such as cross-validation, hyperparameter tuning, and model generalization.
