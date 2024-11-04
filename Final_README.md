
# DTSE Data Engineer Assignment Documentation

## 1. Introduction
This assignment consists of creating an automated solution for processing real estate data from a CSV file. CSV file is processed and transformed into required format. In final part we store the results in database file which we can work with SQL.

## 2. Project Structure
The files and directories in this project are organized to facilitate data handling, result storing and running predictions:

- `main.py` - main script that handasddsles data loading, processing, and storing predictions.
- `housing.csv` - source file containing real estate data. (from assignment)
- `model.joblib` - file with the pre-trained model. (from assignment)
- `housing_data.db` - SQLite database for storing transformed data and predictions. (DB file with 3 tables)
- `requirements.txt` - list of required packages to run the project. (All the packages are copied from original assignment and new ones are part of Python Standard library 
- `assingment.md` - original assignment.
- `Final_README.md` - This markdown file which is used as final description of project.


## 3. Installation and Setup
Clone the project or copy all necessary files to a local directory.

Use the model provided in the model.joblib file.
Python 3.9.13 was tested with the solution, thus this version is recommended to use.

Install the required packages from `requirements.txt` using the following command:

```bash
pip install -r requirements.txt
```

Solution was developed in Pycharm 2024.2.2 and the final result (DB file and test queries) were viewed and in DB browser (SQLite)

## 4. Data Processing

### 4.1 Data Cleaning and Preprocessing
- Load data from `housing.csv`
- CSV file is transformed into required format and loaded as pandas Dataframe. All the necesarry transformations before the predicitng are in prepare_data function (eliminating NA values, some rows which cannot be fitted into model, renaming columns)
- splitting Dataframe into test and train sets by 0.2 (20 %)
- Encode categories in the `ocean_proximity` column using one-hot encoding for model compatibility.

### 4.2 Database Storage
- Transformed data is stored in the SQLite database `housing_data.db` in the `transformed_data` table.
- Predictions are stored in the `predictions` table, where each row contains the record ID, actual value, and predicted value.
- Test input is also stored in database as a 1-row table in `test_predictions`
- `transformed_data` and `predictions` can be joined together as 1:1 on ID key
- Example of SQL query used in further analysis - `Select * from transformed_data t, predictions p where 1=1 and t.id=p.id;`

## 5. Model Prediction
- The model is loaded from the `model.joblib` file and used to make predictions on test data.
- Predictions are split as follows:
  - **Training Set** - used to evaluate the model's accuracy.
  - **Test Set** - results from the test set are compared with actual values using the Mean Absolute Percentage Error (MAPE) metric.

### Function: create_test_data
This function generates a dictionary containing example input values for the model.
We can give different arguments (input1,input2,input3) for testing the input data

## 6. Usage Example

The script will perform the following steps:

1. Load and preprocess data.
2. Store transformed data in the database.
3. Load the pre-trained model and make predictions on the test data.
4. Calculate MAPE for both the training and test sets. (the lower the better)
5. Save predictions to the database.

### Results
The output values and predictions are saved in the `transformed_data`, `predictions`, and `test_predictions` tables within the `housing_data.db` database. Also all the necessary outputs and logs are displayed on console. 

## 7. Error Handling
The following error handling is implemented:

- `FileNotFoundError`: If the input or model file cannot be found.
- `EmptyDataError`: Handling for an empty file.
- `ParserError`: For errors parsing input data.
- `sqlite3.Error`: For database-related errors.
- `Exception` : For other reasons

Errors are used in almost all the functions and could be extended into more types, depending on the requirments of the user. Simultaneously with errors there are implemented exit of process with code 1.

## 8. Optional Extensions

### 8.1 Logging
Logging is configured to output information to console. For the production it would be the best to output each interation of script to make a log file which captures all the information into separate file for the best visibility. Good practice is using structured logging with JSON files and adding contextual information (ID, session ID, request ID)

### 8.2 Testing
Basic checks are implemented for model loading and data preprocessing. Testing of input data is also implemented ( 3 different inputs - can be configured in main function). Next thing would be to add testing of each funtion.a It would allow for quick identification of bugs when changes are made. We could use `pytest` for writing these tests.

### 8.3 APIs
We can implement APIs to fetch real-time data from external sources, this would eliminate dealing with local CSV file in our example. Most web-sites/servers have their APIs code posted. Another solution would be giving input data as Post request (I would recommend Postman)

## 9. Conclusion
This project implements an automated solution for processing, predicting, and storing real estate data using Python, SQLite, and a pre-trained model.
