import sys
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import logging


# Configure logging to output debug information to the console
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# File paths and configuration constants
TRAIN_DATA = 'housing.csv'  # Path to the training data CSV file
MODEL_NAME = 'model.joblib'  # Path to the pre-trained model file
DATABASE_NAME = 'housing_data.db'  # Path to the SQLite database
RANDOM_STATE = 100  # Random seed for reproducibility


# Function to load, clean, and preprocess the data
def prepare_data(input_data_path):
    # Load data from CSV file
    try:
        df = pd.read_csv(input_data_path)
    except FileNotFoundError:
        logging.error(f"File '{input_data_path}' not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logging.error(f"The file '{input_data_path}' is empty.")
        sys.exit(1)
    except pd.errors.ParserError:
        logging.error(f"Error parsing the file '{input_data_path}'.")
        sys.exit(1)
    # Remove rows with any null values
    df = df.dropna()
    # Filter out rows with the string 'Null' (in case of string type values)
    df = df[~df.astype(str).apply(lambda row: row.str.contains('Null').any(), axis=1)]
    # Standardize column names
    df = df.rename(columns={'LAT': 'LATITUDE', 'MEDIAN_AGE': 'HOUSING_MEDIAN_AGE',
                            'ROOMS': 'TOTAL_ROOMS', 'BEDROOMS': 'TOTAL_BEDROOMS', 'POP': 'POPULATION'})
    df.columns = df.columns.str.lower()  # Convert all column names to lowercase

    # Encode categorical column 'ocean_proximity' with one-hot encoding
    categorical_cols = ['ocean_proximity']
    df = pd.get_dummies(df, columns=categorical_cols)

    # Add a unique identifier if not already present
    if 'id' not in df.columns:
        df['id'] = range(1, len(df) + 1)

    # Separate features (X) and target (y)
    df_features = df.drop(['median_house_value'], axis=1)
    y = df['median_house_value'].values

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df_features, y, test_size=0.2, random_state=RANDOM_STATE)

    return X_train, X_test, y_train, y_test, df

def create_test_data(test_case='input1'):
    if test_case == 'input1':
        return {
            'longitude': [-122.64],
            'latitude': [38.01],
            'housing_median_age': [36.0],
            'total_rooms': [1336.0],
            'total_bedrooms': [258.0],
            'population': [678.0],
            'households': [249.0],
            'median_income': [5.5789],
            'ocean_proximity': ['NEAR OCEAN']
        }
    elif test_case == 'input2':
        return {
            'longitude': [-115.73],
            'latitude': [33.35],
            'housing_median_age': [23.0],
            'total_rooms': [1586.0],
            'total_bedrooms': [448.0],
            'population': [338.0],
            'households': [182.0],
            'median_income': [1.2132],
            'ocean_proximity': ['INLAND']
        }
    elif test_case == 'input3':
        return {
            'longitude': [-117.96],
            'latitude': [33.89],
            'housing_median_age': [24.0],
            'total_rooms': [1332.0],
            'total_bedrooms': [252.0],
            'population': [625.0],
            'households': [230.0],
            'median_income': [4.4375],
            'ocean_proximity': ['<1H OCEAN']
        }
    else:
        raise ValueError("Unknown test case provided.")



# Function to save data to an SQLite database
def save_to_db(df, table_name, conn):
    try:
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        logging.info(f'Table: {table_name} successfully saved to database.')
    except sqlite3.Error as e:
        logging.error(f"Error saving to database table '{table_name}': {e}")


# Function to display a database table's contents in the log output
def display_table(table_name, conn):
    query = f"SELECT * FROM {table_name};"  # SQL query to select all data
    table_data = pd.read_sql_query(query, conn)  # Read table into DataFrame
    logging.info(f"Contents of table '{table_name}':\n{table_data}")


# Function to load model and make predictions
def predict(X, model):
    X = X.reindex(columns=model.feature_names_in_, fill_value=0)  # Match model's expected columns
    Y = model.predict(X)  # Make predictions
    return Y


# Function to load a pre-trained model from a file
def load_model(filename):
    try:
        model = joblib.load(filename)
        return model
    except FileNotFoundError:
        logging.error(f"Model file '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        sys.exit(1)


def calculate_mape(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # in %
    return mape


# Main script execution
if __name__ == '__main__':
    try:
        logging.info('Preparing the data...')
        # Load and process the data
        X_train, X_test, y_train, y_test, full_data = prepare_data(TRAIN_DATA)

        # Connect to the SQLite database
        conn = sqlite3.connect(DATABASE_NAME)

        # Save processed data to database table 'transformed_data'
        save_to_db(full_data, 'transformed_data', conn)

        # Load pre-trained model and generate predictions on the test dataset
        logging.info('Loading the model...')
        model = load_model(MODEL_NAME)
        logging.info('Calculating train dataset predictions...')
        y_pred_train = predict(X_train, model)  # Predictions on training set
        logging.info('Calculating test dataset predictions...')
        y_pred_test = predict(X_test, model)  # Predictions on testing set

        # Save predictions to database table 'predictions' with identifier
        predictions_df = pd.DataFrame({
            'id': X_test['id'],  # Include identifier from transformed data
            'Actual': y_test,
            'Predicted': y_pred_test
        })
        save_to_db(predictions_df, 'predictions', conn)

        # Test predictions with specific input data
        input_data = pd.DataFrame(create_test_data())

        # Encode 'ocean_proximity' column in the input data
        ocean_dummies = pd.get_dummies(input_data['ocean_proximity'], prefix='ocean_proximity')
        input_data = pd.concat([input_data, ocean_dummies], axis=1)
        input_data.drop(columns=['ocean_proximity'], inplace=True)

        # Add any missing columns as zeros to match model input format
        for col in model.feature_names_in_:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[model.feature_names_in_]

        # Make a prediction with test input data
        prediction = model.predict(input_data)
        logging.info(f'Predicted house price for test input: {prediction[0]}')

        # Save test input data and prediction to database table 'test_predictions'
        test_data_df = input_data.copy()
        test_data_df['Predicted_Price'] = prediction
        save_to_db(test_data_df, 'test_predictions', conn)

        mape_train = calculate_mape(y_train, y_pred_train)
        mape_test = calculate_mape(y_test, y_pred_test)

        # MAPE output
        logging.info(f"Mean Absolute Percentage Error (MAPE) on training data: {mape_train:.2f}%")
        logging.info(f"Mean Absolute Percentage Error (MAPE) on test data: {mape_test:.2f}%")

        # Display contents of all tables in the database

        display_table('transformed_data', conn)
        display_table('predictions', conn)
        display_table('test_predictions', conn)


    except Exception as e:
        logging.error(f"An error in main block occurred: {e}")
    finally:
        # Close the database connection
        if conn is not None:
            conn.close()  # Ensure connection is closed if it was opened
            logging.info("Database connection closed.")

    logging.info("Data processing and prediction storage completed.")