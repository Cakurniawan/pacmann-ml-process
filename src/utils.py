import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Function named read_data 
def load_data(fname):
    """ 

    This function for import dataset from specific directory in the local computer 
    into pandas dataframe, for later to be trained into a Machine Learning Model.

    Parameter:
    - fname (string) = The Filename that we want to import and transform into dataframe.

    Return:
    - data (Dataframe) = a pandas Dataframe from import .csv file.

    Candra Kurniawan | Pacmann AI 2024.
    
    """
    # Import the data from local computer into dataframe
    # mencari file dengan kata belakang .xlsx
    if fname.endswith('.xlsx'):
        data = pd.read_excel(fname)

    # mencari file dengan kata belakang .csv
    elif fname.endswith('.csv'):
        data = pd.read_csv(fname, sep = ',')

    # jika tidak ada file xlsx dan csv maka akan Raise Error
    else:
        raise ValueError(f"File tidak ditemukan {fname} harus berbentuk .xlsx atau .csv")
    
    print(f"Data Shape: {data.shape}")

    return data

# SPLIT INPUT OUTPUT FUNCTION
def split_input_output(data, target_col):
    """
    This function for splitting data into input for train data (X) 
    and output for target/predict data (y).

    This function for splitting data into input for train data (X) 
    and output for target/predict data (y).

    Parameter : -> has two parameter
                1. data (pd.DataFrame)
                2. target_col (column pandas)
            -> Print the data.shape after splitting
            -> Then, Returning the value of X and y

    Returning : -> Data input that will be trained further in (X)
                -> And data target output that will be performed analysis/predict in (y).

    """
    # Splitting the data into input (X) and output (y)
    X = data.drop(target_col, axis = 1)
    y = data[target_col]

    # Print the shape of the data after splitting
    print(f"Original data shape: {data.shape}")
    print(f"X data shape: {X.shape}")
    print(f"y data shape: {y.shape}")

    return X, y

# Function Split into Train, Test, valid data
def split_train_test(X, y, test_size, random_state):
    """  
    This function for splitting the input and output data into
    Training, validation, and test dataset.

    Parameter : -> Has four Parameters:
                1. X = the input (pd.DataFrame)
                2. y = the output (pd.DataFrame)
                3. test_size = the test size between 0 - 1 (float)
                4. seed = the random state (int)
                5. stratify = This arguments is used for representative our 
                    imbalance output dataset. we set it (y) the output data.
            -> Print the data shape after splitting
            -> Then return the X_train, X_test, y_train, y_test

    Returning : X_train -> data input as (X) for training data
                X_test -> data input as (X) for test data
                y_train -> data output as (y) for training data
                y_test -> data output as (y) for test data
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = test_size, 
                                                        random_state = random_state,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test

# Function Serialize Data
def serialize_data(data, path):
    """This function for serialize data, meaning of Serialize data for export.
    the Dataset/Model into Binary Data (Pickling) or Python Object (Unpickling)

    Parameter : -> Has two Parameter:
                1. data = the object that want to Serialize (Dataset/Model)
                2. path = the target path for placing or placed .pkl
                
            -> dump the object data into .pkl

    Returning : serialize -> the data that finish pickling into .pkl file
    
                X_train.pkl -> data input as (X) for training data
                X_test.pkl -> data input as (X) for test data
                X_valid.pkl -> data input as (X) for valid data
                y_train.pkl -> data output as (y) for training data
                y_test.pkl -> data output as (y) for test data
                y_valid.pkl -> data output as (y) for valid data

    """

    # Save the trained model to a file
    serialize = joblib.dump(data, path)

    return serialize

# Function Deserialize Data
def deserialize_data(path):
    """This function for deserialize data, meaning of unSerialize data is for import
    the Binary Object into the Python Object (Unpickling) it can be a Dataset or Model ML.

    Parameter : -> Has one Parameter:
                1. path = the target path for placing or placed .pkl
                
            -> import the object data into python data variable.

    Returning : deserialize -> a variable that contain the data.

    """

    # Save the trained model to a file
    data = joblib.load(path)

    return data

