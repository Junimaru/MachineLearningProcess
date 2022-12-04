import pandas as pd
import util as utils
import copy
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer

def read_raw_data(config: dict) -> pd.DataFrame:
    # Return raw dataset
    return pd.read_csv(config["dataset_path"])

def check_data(input_data: pd.DataFrame, config: dict, api: bool = False):
    input_data = copy.deepcopy(input_data)
    config = copy.deepcopy(config)
    
    # Check column data types
    assert input_data.select_dtypes("float").columns.to_list() == \
        config["float_columns"], "an error occurs in float column(s)."    
    assert input_data.select_dtypes("int").columns.to_list() == \
        config["int_columns"], "an error occurs in int column(s)."

def impute_data(input_data: pd.DataFrame, config: dict):
    imputer = KNNImputer(n_neighbors=10, weights="uniform")
    l = imputer.fit_transform(input_data)
    
    return pd.DataFrame(l,columns=input_data.columns)      

def split_data(raw_data_impute: pd.DataFrame, config: dict):
    # Split predictor and label
    x = raw_data_impute[config["predictors"]].copy()
    y = raw_data_impute[config["label"]].copy()

    # 1st split train and test
    x_train, x_test, \
    y_train, y_test = train_test_split(
        x, y,
        test_size = config["test_size"],
        random_state = 42,
        stratify = y
    )

    # 2nd split test and valid
    x_valid, x_test, \
    y_valid, y_test = train_test_split(
        x_test, y_test,
        test_size = config["valid_size"],
        random_state = 42,
        stratify = y_test
    )

    return x_train, x_valid, x_test, y_train, y_valid, y_test


if __name__ == "__main__":
    # 1. Load configuration file
    config = utils.load_config()

    # 2. Read all raw dataset
    raw_dataset = read_raw_data(config)

    # 3. Data defense for non API data
    check_data(raw_dataset, config)

    # 4. Impute data

    raw_data_impute = impute_data(raw_dataset, config)

    # 4. Splitting train, valid, and test set
    x_train, x_valid, x_test, \
        y_train, y_valid, y_test = split_data(raw_data_impute, config)

    # 5. Save train, valid and test set
    utils.pickle_dump(x_train, config["train_set_path"][0])
    utils.pickle_dump(y_train, config["train_set_path"][1])

    utils.pickle_dump(x_valid, config["valid_set_path"][0])
    utils.pickle_dump(y_valid, config["valid_set_path"][1])

    utils.pickle_dump(x_test, config["test_set_path"][0])
    utils.pickle_dump(y_test, config["test_set_path"][1])

    utils.pickle_dump(raw_dataset, config["dataset_cleaned_path"])