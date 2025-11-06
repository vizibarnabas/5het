import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the parent directory to sys.path so we can import MLModel
current_file_path = Path(__file__).resolve()  # Get the path of the current file
parent_directory = current_file_path.parent.parent  # Get the parent directory (two levels up)
sys.path.append(str(parent_directory))  # Add the parent directory to sys.path

from MLModel import MLModel

def test_prediction_accuracy():
    obj_mlmodel = MLModel()  
    data_path = 'data/horse-colic-train.csv'
    df = pd.read_csv(data_path)

    # Training pipeline
    df_preprocessed = obj_mlmodel.preprocessing_pipeline(df)
    y_expected = df_preprocessed['V24']
    accuracy_train_pipeline_full = obj_mlmodel.get_accuracy_full(
                                            df_preprocessed.drop(columns="V24"), 
                                            y_expected)
    accuracy_train_pipeline_full = np.round(accuracy_train_pipeline_full, 2)

    # Inference pipeline
    obj_mlmodel = MLModel()  
    df = pd.read_csv(data_path)

    preprocessed_list = []
    for i in df.iterrows():
        preprocessed_list.append(obj_mlmodel.preprocessing_pipeline_inference(i[1]))

    df_preprocessed = pd.concat(preprocessed_list)
    accuracy_inference_pipeline_full = obj_mlmodel.get_accuracy_full(df_preprocessed, y_expected)
    accuracy_inference_pipeline_full = np.round(accuracy_inference_pipeline_full, 2)
    print(accuracy_train_pipeline_full, accuracy_inference_pipeline_full)

    assert accuracy_train_pipeline_full == accuracy_inference_pipeline_full, 'Inference prediction accuracy is not as expected'