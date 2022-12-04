import data_pipeline
import util as utils
import pandas as pd
import numpy as np

def test_check_data():
    # Arrange
    config = utils.load_config()
    mock_data = {'ph': [8.31],
        'Hardness': [214.37],
        'Solids': [22018.41],
        'Chloramines': [8.05],
        'Sulfate': [356.88],
        'Conductivity': [363.26],
        'Organic_carbon': [18.43],
        'Trihalomethanes': [100.34],
        'Turbidity': [4.62],
        'Potability': [0]}
    mock_data = pd.DataFrame(mock_data)
    expected_data = {'ph': [8.31],
        'Hardness': [214.37],
        'Solids': [22018.41],
        'Chloramines': [8.05],
        'Sulfate': [356.88],
        'Conductivity': [363.26],
        'Organic_carbon': [18.43],
        'Trihalomethanes': [100.34],
        'Turbidity': [4.62],
        'Potability': [0]}
    expected_data = pd.DataFrame(expected_data)

    # Act
    processed_data = data_pipeline.check_data(mock_data, config)

    # Assert
    assert processed_data.equals(expected_data)