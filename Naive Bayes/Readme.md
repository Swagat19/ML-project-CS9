---
# Naive Bayes Model
This repository contains Jupyter notebooks for implementing a Naive Bayes model from scratch for classification tasks. The repository includes two main notebooks: `NaiveBayes_EDA.ipynb` for exploratory data analysis (EDA) and data preprocessing, and `NaiveBayes_scratch.ipynb` for implementing and evaluating the Naive Bayes model.

## Naive Bayes Overview

The Naive Bayes model is a family of probabilistic classifiers based on Bayes' Theorem. It is primarily used for classification tasks, particularly in applications such as text classification, spam detection, sentiment analysis, and document categorization. The "naive" part of the name refers to the model's assumption that the features are independent of each other, which simplifies the computation. Despite this assumption, Naive Bayes performs surprisingly well in many real-world situations.

### How It Works

Naive Bayes classifiers work by estimating the likelihood of each class given the feature values and then applying Bayes' Theorem to compute the posterior probabilities. The model selects the class with the highest posterior probability as the predicted output. Naive Bayes is computationally efficient and works well with large datasets.

Key steps involved in Naive Bayes classification:
1. **Training**: Calculate the prior probabilities of each class and the conditional probabilities of each feature given the class.
2. **Prediction**: Use the prior and conditional probabilities to compute the posterior probability for each class given the input features and predict the class with the highest posterior probability.
3. **Assumptions**: The features are conditionally independent, and each feature contributes equally to the outcome.

The formula for Bayes' Theorem is:

**P(C|X) = [P(X|C) * P(C)] / P(X)**

Where:
- **P(C|X)** is the posterior probability of class **C** given the feature set **X**.
- **P(X|C)** is the likelihood of the feature set **X** given class **C**.
- **P(C)** is the prior probability of class **C**.
- **P(X)** is the marginal probability of the feature set **X**.

Naive Bayes assumes that the features are conditionally independent, which simplifies the likelihood calculation as:

**P(X|C) = P(x1|C) * P(x2|C) * ... * P(xn|C)**

Where **x1, x2, ..., xn** are the individual features.

### Laplace Smoothing

In cases where a feature category does not appear in the training data for a particular class, the probability assigned to that category can be zero, which can negatively impact predictions. To address this, **Laplace smoothing** is applied.

The formula for Laplace smoothing is:

**P(xi|C) = (count(xi in C) + 1) / (count(C) + |X|)**

Where:
- **count(xi in C)** is the frequency of feature **xi** in class **C**.
- **count(C)** is the total number of instances in class **C**.
- **|X|** is the total number of unique feature values (categories).

Laplace smoothing ensures that every feature has a non-zero probability, preventing zero-frequency problems in the model.

## Repository Contents

This repository includes two main notebooks for building and evaluating the Naive Bayes model from scratch:

1. **`NaiveBayes_EDA.ipynb`**: This notebook covers the exploratory data analysis (EDA), data preprocessing, and encoding steps needed to prepare the dataset for the Naive Bayes model.
2. **`NaiveBayes_scratch.ipynb`**: This notebook contains the implementation of the Naive Bayes model from scratch, along with a comparison of performance between the custom implementation and an implementation using the `scikit-learn` library.

## Installation

Ensure you have the necessary libraries installed. You can install them using the following command:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

You can download the zip file of the repository or clone the repository using the following command:

```bash
git clone https://github.com/Swagat19/ML-project-CS9.git
```

### 1. NaiveBayes_EDA.ipynb

- This notebook includes exploratory data analysis (EDA), data preprocessing, and encoding of categorical variables to prepare the dataset for Naive Bayes classification.
- To run this notebook, ensure you have the necessary libraries installed as mentioned above.
- The dataset can be changed by uncommenting the appropriate line in the notebook:
  ```python
  data = pd.read_csv('CSV_Files/SWAGAT_INPUT_income_dataset_balanced.csv')
  # data = pd.read_csv('CSV_Files/InputData.csv')
  # data = pd.read_csv('CSV_Files/BalancedInputData.csv')
  ```
- Run this notebook to explore the dataset and prepare it for model training.

### 2. NaiveBayes_scratch.ipynb

- This notebook contains the full implementation of the Naive Bayes model from scratch, along with a comparison with the `scikit-learn` implementation of Naive Bayes.
- Before running this notebook, ensure you have completed the EDA and preprocessing steps in `NaiveBayes_EDA.ipynb`.
- Run this notebook to train the Naive Bayes model and evaluate its performance on the processed dataset.

## Running the Notebooks

- It is recommended to run the notebooks in Jupyter Notebook, JupyterLab, or Visual Studio Code.
- Follow this sequence for execution:
  1. Run the `NaiveBayes_EDA.ipynb` notebook to perform data preprocessing and encoding.
  2. Once the EDA notebook has been executed, proceed to the `NaiveBayes_scratch.ipynb` notebook to implement the Naive Bayes model and evaluate its performance.
- You can execute the entire notebook at once or step through the cells interactively.

## Notes

- Ensure that the dataset paths in both notebooks (`NaiveBayes_EDA.ipynb` and `NaiveBayes_scratch.ipynb`) are correctly set according to your environment.
- The notebook assumes that the dataset has been preprocessed and encoded as per the EDA notebook's output. If you modify the preprocessing or encoding steps, you may need to adjust the `NaiveBayes_scratch.ipynb` notebook accordingly.

## Applications of Naive Bayes

Naive Bayes is particularly useful in the following areas:
- **Text Classification**: Commonly used in spam detection, sentiment analysis, and document classification.
- **Medical Diagnosis**: Helps in predicting the probability of diseases based on symptoms.
- **Recommendation Systems**: Predicts user preferences based on past data.
- **Real-time Predictions**: Due to its fast performance, it is ideal for making real-time predictions in large-scale applications.

### Advantages
- **Simplicity**: Easy to implement and interpret.
- **Scalability**: Handles large datasets efficiently.
- **Speed**: Fast in both training and prediction phases.
- **Works well with categorical data**: Especially effective for text classification problems.

### Limitations
- **Independence Assumption**: The assumption that all features are independent may not hold in all cases, leading to suboptimal performance in some tasks.
- **Zero Frequency**: If a category in a feature is not present in the training data, the model assigns it a zero probability, which may affect predictions. This can be mitigated using techniques like Laplace smoothing.

## Conclusion

This repository provides a hands-on approach to implementing the Naive Bayes model from scratch, covering the entire workflow from data exploration and preprocessing to model implementation and evaluation. The Naive Bayes classifier, despite its simplicity, is a powerful and widely used algorithm for a variety of classification tasks.
