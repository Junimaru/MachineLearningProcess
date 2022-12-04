import pandas as pd
import util as utils
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

def load_dataset(config_data: dict):
    x_train = utils.pickle_load(config_data["train_set_path"][0])
    y_train = utils.pickle_load(config_data["train_set_path"][1])

    x_valid = utils.pickle_load(config_data["valid_set_path"][0])
    y_valid = utils.pickle_load(config_data["valid_set_path"][1])

    x_test = utils.pickle_load(config_data["test_set_path"][0])
    y_test = utils.pickle_load(config_data["test_set_path"][1])

    train_set = pd.concat([x_train, y_train], axis = 1)
    valid_set = pd.concat([x_valid, y_valid], axis = 1)
    test_set = pd.concat([x_test, y_test], axis = 1)

    return train_set, valid_set, test_set

def ros_fit_resample(set_data, config):
    x_ros, y_ros = SMOTE(random_state = 42).fit_resample(
        set_data.drop(columns = config["label"]),
        set_data[config["label"]]
    )
    return pd.concat([x_ros, y_ros], axis = 1)


def scale_data(set_data, config):
    set_data = set_data.copy()
    scaler = MinMaxScaler()
    x_std = set_data.drop(columns = config["label"])
    y_std = set_data[config["label"]]
    names = x_std.columns
    d = scaler.fit_transform(x_std)
    set_data1 = pd.DataFrame(d, columns=names)
    return pd.concat([set_data1,y_std], axis = 1)

if __name__ == "__main__":
    # 1. Load configuration file
    config = utils.load_config()

    # 2. Load dataset
    train_set, valid_set, test_set = load_dataset(config)

    # 3. Oversample dataset
    train_set_bal = ros_fit_resample(train_set, config)

    # 4. Scalling data
    train_set_bal_cleaned = scale_data(train_set_bal, config)

    # 5. Dump set data
    utils.pickle_dump(
            train_set_bal_cleaned[config["predictors"]],
            config["train_feng_set_path"][0]
    )
    utils.pickle_dump(
            train_set_bal_cleaned[config["label"]],
            config["train_feng_set_path"][1]
    )


    utils.pickle_dump(
            valid_set[config["predictors"]],
            config["valid_feng_set_path"][0]
    )
    utils.pickle_dump(
            valid_set[config["label"]],
            config["valid_feng_set_path"][1]
    )


    utils.pickle_dump(
            test_set[config["predictors"]],
            config["test_feng_set_path"][0]
    )
    utils.pickle_dump(
            test_set[config["label"]],
            config["test_feng_set_path"][1]
    )

    