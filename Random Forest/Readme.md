# Random Forest Model

This repository contains Jupyter notebooks for implementing a Random Forest Model from scratch for classification tasks. This repository contains 4 Jupyter notebooks: `Filter by income.ipynb` ,  `input visualize.ipynb` , `Random Forest Implementation.ipynb` and `Random forest with external library .ipynb`.


## Note

- To simulate result please comment the earlier file input and uncomment the file which you want to take data from in first cell.
- Ensure that the input data file path in all notebooks is correctly set to the location of your dataset.
- All of our input data is in `CsvFile` folder and it needs to kept in same directory where the notebooks are stored and run.
- In case if any library is not installed then install it to remove errors.


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

## 1. Random Forest Implementation.ipynb

This notebook contains implementing Tree classifier and ensembling results of trees to form a random forest model.
Run this for different input data present in Csv Files folder to simulate results in report. 

## 2. Random forest with external library.ipynb

This notebook contains random forest model using external library.
This notebook is to check how well our implementation works comparing to external library
Run this for different input data present in Csv Files folder to simulate results in report. 

## 3. Filter by income.ipynb

This notebook takes input data and filter the data to remove the outliers and also balanced new data by under sampling dominant class.

## 4. input visualize.ipynb

This notebook is to visualize how our data looks like after we filter our input data.
