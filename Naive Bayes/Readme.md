# Naive Bayes Model

This repository contains Jupyter notebooks for implementing a Naive Bayes model from scratch for classification tasks. This repository contains two Jupyter notebooks: `NaiveBayes_EDA.ipynb` and `NaiveBayes_scratch.ipynb`.

## Installation

Ensure you have the necessary libraries installed. You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

Download the zip file of the repository or clone the repository:
```bash
git clone https://github.com/Swagat19/ML-project-CS9.git
```

## 1. NaiveBayes_EDA.ipynb

- This notebook contains the exploratory data analysis (EDA), preprocessing, and encoding steps for the Naive Bayes model.
- To run this notebook, ensure you have the necessary libraries installed. You can install them using pip:
  ```
  pip install pandas numpy matplotlib seaborn
  ```
- You can change the input data by uncommenting the appropriate line in the notebook:
  ```
  data = pd.read_csv('CSV_Files/SWAGAT_INPUT_income_dataset_balanced.csv')
  # data = pd.read_csv('CSV_Files/InputData.csv')
  # data = pd.read_csv('CSV_Files/BalancedInputData.csv')
  ```
- Run the notebook to perform EDA and prepare the data for the Naive Bayes model.

## 2. NaiveBayes_scratch.ipynb

- This notebook contains the implementation of the Naive Bayes model from scratch, as well as running the model using an external library and our implementation.
- Before running this notebook, ensure you have the necessary libraries installed. You can install them using pip:
  ```
  pip install scikit-learn
  ```
- Run the notebook after running the EDA notebook. The data processed in the EDA notebook will be used for model training and evaluation.

## Running the Notebooks

- It is recommended to run the two files in Jupyter Notebook or Visual Studio Code.
- Run the EDA notebook (`NaiveBayes_EDA.ipynb`) first to perform preprocessing and encoding.
- Once the EDA notebook has been run, you can run the `NaiveBayes_scratch.ipynb` notebook to implement the Naive Bayes model and evaluate its performance.
- You can run the complete notebook in one go or step through it cell by cell.

## Note

- Ensure that the input data file path in both notebooks (`NaiveBayes_EDA.ipynb` and `NaiveBayes_scratch.ipynb`) is correctly set to the location of your dataset.
- The notebook assumes that the dataset has been preprocessed and encoded as per the EDA notebook's output. If you make changes to the preprocessing or encoding steps, you may need to modify the `NaiveBayes_scratch.ipynb` notebook accordingly.

