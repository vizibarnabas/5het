
import os
import sys
from six.moves import urllib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import logging
import dill
import mlflow
import mlflow.sklearn

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - %(levelname)s - %(message)s')


# --- Custom Exception ---
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_detail)

    @staticmethod
    def get_detailed_error_message(error_message, error_detail: sys) -> str:
        _, _, exc_tb = error_detail.exc_info()
        line_number = exc_tb.tb_frame.f_lineno
        file_name = exc_tb.tb_frame.f_code.co_filename
        return f"Error occurred in script: [{file_name}] at line number: [{line_number}] error message: [{error_message}]"

    def __str__(self):
        return self.error_message


# --- Configuration Classes (simplified) ---
class DataIngestionConfig:
    def __init__(self, artifact_dir):
        self.root_dir = os.path.join(artifact_dir, "data_ingestion")
        self.dataset_download_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
        self.raw_data_dir = os.path.join(self.root_dir, "raw_data")
        self.ingested_train_dir = os.path.join(self.root_dir, "ingested", "train")
        self.ingested_test_dir = os.path.join(self.root_dir, "ingested", "test")

class DataValidationConfig:
    def __init__(self, artifact_dir):
        self.root_dir = os.path.join(artifact_dir, "data_validation")
        self.schema_file_path = os.path.join(self.root_dir, "schema.yaml") # Example path

class DataTransformationConfig:
    def __init__(self, artifact_dir):
        self.root_dir = os.path.join(artifact_dir, "data_transformation")
        self.transformed_train_dir = os.path.join(self.root_dir, "transformed", "train")
        self.transformed_test_dir = os.path.join(self.root_dir, "transformed", "test")
        self.preprocessed_object_file_path = os.path.join(self.root_dir, "preprocessed", "preprocessed.pkl")

class ModelTrainerConfig:
    def __init__(self, artifact_dir):
        self.root_dir = os.path.join(artifact_dir, "model_trainer")
        self.trained_model_file_path = os.path.join(self.root_dir, "trained_model", "model.pkl")

# --- Artifact Classes (simplified) ---
class DataIngestionArtifact:
    def __init__(self, train_file_path, test_file_path, is_ingested, message):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.is_ingested = is_ingested
        self.message = message

class DataValidationArtifact:
    def __init__(self, is_validated, message, schema_file_path):
        self.is_validated = is_validated
        self.message = message
        self.schema_file_path = schema_file_path

class DataTransformationArtifact:
    def __init__(self, is_transformed, message, transformed_train_file_path, transformed_test_file_path, preprocessed_object_file_path):
        self.is_transformed = is_transformed
        self.message = message
        self.transformed_train_file_path = transformed_train_file_path
        self.transformed_test_file_path = transformed_test_file_path
        self.preprocessed_object_file_path = preprocessed_object_file_path

class ModelTrainerArtifact:
    def __init__(self, is_trained, message, trained_model_file_path, train_accuracy, test_accuracy, model_accuracy):
        self.is_trained = is_trained
        self.message = message
        self.trained_model_file_path = trained_model_file_path
        self.train_accuracy = train_accuracy
        self.test_accuracy = test_accuracy
        self.model_accuracy = model_accuracy


# --- Utility Functions ---
def save_object(file_path: str, obj: object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e

def load_object(file_path: str) -> object:
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e

def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, array)
    except Exception as e:
        raise CustomException(e, sys) from e

def load_numpy_array_data(file_path: str) -> np.ndarray:
    try:
        return np.load(file_path)
    except Exception as e:
        raise CustomException(e, sys) from e


# --- ML Components ---
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self) -> str:
        try:
            os.makedirs(self.config.raw_data_dir, exist_ok=True)
            file_path = os.path.join(self.config.raw_data_dir, "crx.data")
            urllib.request.urlretrieve(self.config.dataset_download_url, file_path)
            return file_path
        except Exception as e:
            raise CustomException(e, sys) from e

    def split_data(self, raw_data_path: str) -> DataIngestionArtifact:
        try:
            col_names = [f'col{i}' for i in range(16)]
            df = pd.read_csv(raw_data_path, header=None, names=col_names)
            df.replace('?', np.nan, inplace=True)
            df.dropna(subset=["col15"], inplace=True)

            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            for train_idx, test_idx in split.split(df, df["col15"]):
                train_set = df.loc[train_idx]
                test_set = df.loc[test_idx]

            os.makedirs(self.config.ingested_train_dir, exist_ok=True)
            os.makedirs(self.config.ingested_test_dir, exist_ok=True)
            
            train_path = os.path.join(self.config.ingested_train_dir, "train.csv")
            test_path = os.path.join(self.config.ingested_test_dir, "test.csv")

            train_set.to_csv(train_path, index=False)
            test_set.to_csv(test_path, index=False)

            return DataIngestionArtifact(train_path, test_path, True, "Data Ingestion Complete")
        except Exception as e:
            raise CustomException(e, sys) from e
            
    def run(self):
        raw_path = self.download_data()
        return self.split_data(raw_path)


class DataValidation:
    def __init__(self, config: DataValidationConfig, ingestion_artifact: DataIngestionArtifact):
        self.config = config
        self.ingestion_artifact = ingestion_artifact

    def validate(self) -> DataValidationArtifact:
        # Basic validation: check if files exist
        train_exists = os.path.exists(self.ingestion_artifact.train_file_path)
        test_exists = os.path.exists(self.ingestion_artifact.test_file_path)
        
        if not (train_exists and test_exists):
            raise Exception("Train or test file is missing!")
            
        # More complex validation (schema, drift) would go here
        
        return DataValidationArtifact(True, "Validation Complete", self.config.schema_file_path)

    def run(self):
        return self.validate()

class DataTransformation:
    def __init__(self, config: DataTransformationConfig, ingestion_artifact: DataIngestionArtifact):
        self.config = config
        self.ingestion_artifact = ingestion_artifact

    def get_preprocessor(self) -> ColumnTransformer:
        numerical_cols = ["col1", "col2", "col7", "col10", "col13", "col14"]
        categorical_cols = ["col0", "col3", "col4", "col5", "col6", "col8", "col9", "col11", "col12"]

        num_pipeline = SklearnPipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        cat_pipeline = SklearnPipeline(steps=[
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('one_hot', OneHotEncoder(handle_unknown='ignore')),
            ('scaler', StandardScaler(with_mean=False))
        ])
        
        return ColumnTransformer([
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols)
        ])

    def transform(self) -> DataTransformationArtifact:
        try:
            preprocessor = self.get_preprocessor()
            
            train_df = pd.read_csv(self.ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.ingestion_artifact.test_file_path)

            # Clean and prepare data
            for df in [train_df, test_df]:
                df.replace('?', np.nan, inplace=True)
                for col in ["col1", "col2", "col7", "col10", "col13", "col14"]:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df["col15"] = df["col15"].apply(lambda x: 1 if x == '+' else 0)

            X_train = train_df.drop("col15", axis=1)
            y_train = train_df["col15"]
            X_test = test_df.drop("col15", axis=1)
            y_test = test_df["col15"]

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            if hasattr(X_train_transformed, "toarray"):
                X_train_transformed = X_train_transformed.toarray()
                X_test_transformed = X_test_transformed.toarray()
            
            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]
            
            # Save transformed data
            os.makedirs(self.config.transformed_train_dir, exist_ok=True)
            os.makedirs(self.config.transformed_test_dir, exist_ok=True)
            train_npy_path = os.path.join(self.config.transformed_train_dir, "train.npy")
            test_npy_path = os.path.join(self.config.transformed_test_dir, "test.npy")
            save_numpy_array_data(train_npy_path, train_arr)
            save_numpy_array_data(test_npy_path, test_arr)
            
            # Save preprocessor
            save_object(self.config.preprocessed_object_file_path, preprocessor)

            return DataTransformationArtifact(True, "Transformation Complete", train_npy_path, test_npy_path, self.config.preprocessed_object_file_path)

        except Exception as e:
            raise CustomException(e, sys) from e

    def run(self):
        return self.transform()


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, transformation_artifact: DataTransformationArtifact):
        self.config = config
        self.transformation_artifact = transformation_artifact

    def train(self) -> ModelTrainerArtifact:
        try:
            train_arr = load_numpy_array_data(self.transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.transformation_artifact.transformed_test_file_path)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            model = LogisticRegression()
            param_grid = {'tol': [0.01, 0.001], 'max_iter': [100, 200]} # Simplified grid
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            y_train_pred = best_model.predict(X_train)
            train_acc = accuracy_score(y_train, y_train_pred)
            y_test_pred = best_model.predict(X_test)
            test_acc = accuracy_score(y_test, y_test_pred)

            if abs(train_acc - test_acc) > 0.1: # Overfitting check
                logging.warning("Model might be overfitting.")

            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("test_accuracy", test_acc)

            mlflow.sklearn.log_model(best_model, "model")
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"

            return ModelTrainerArtifact(True, "Training Complete", model_uri, train_acc, test_acc, test_acc)

        except Exception as e:
            raise CustomException(e, sys) from e
            
    def run(self):
        return self.train()


# --- Main Pipeline ---
class Pipeline:
    def __init__(self, artifact_dir="artifacts"):
        self.artifact_dir = artifact_dir
        self.data_ingestion_config = DataIngestionConfig(artifact_dir)
        self.data_validation_config = DataValidationConfig(artifact_dir)
        self.data_transformation_config = DataTransformationConfig(artifact_dir)
        self.model_trainer_config = ModelTrainerConfig(artifact_dir)

    def run(self):
        try:
            logging.info("--- Starting Pipeline ---")
            
            with mlflow.start_run() as run:
                logging.info(f"MLflow run started with ID: {run.info.run_id}")

                # Data Ingestion
                ingestion = DataIngestion(self.data_ingestion_config)
                ingestion_artifact = ingestion.run()
                logging.info(f"Data Ingestion Artifact: {ingestion_artifact.message}")
                mlflow.log_artifact(ingestion_artifact.train_file_path, "ingested_data")
                mlflow.log_artifact(ingestion_artifact.test_file_path, "ingested_data")

                # Data Validation
                validation = DataValidation(self.data_validation_config, ingestion_artifact)
                validation_artifact = validation.run()
                logging.info(f"Data Validation Artifact: {validation_artifact.message}")
                if os.path.exists(validation_artifact.schema_file_path):
                    mlflow.log_artifact(validation_artifact.schema_file_path, "validation")

                # Data Transformation
                transformation = DataTransformation(self.data_transformation_config, ingestion_artifact)
                transformation_artifact = transformation.run()
                logging.info(f"Data Transformation Artifact: {transformation_artifact.message}")
                mlflow.log_artifact(transformation_artifact.preprocessed_object_file_path, "preprocessor")

                # Model Trainer
                trainer = ModelTrainer(self.model_trainer_config, transformation_artifact)
                trainer_artifact = trainer.run()
                logging.info(f"Model Trainer Artifact: {trainer_artifact.message}")

                # Register the model
                mlflow.register_model(
                    model_uri=trainer_artifact.trained_model_file_path,
                    name="CreditApprovalModel"
                )

            logging.info("--- Pipeline Finished ---")

        except Exception as e:
            logging.error(f"Pipeline failed: {e}")
            raise CustomException(e, sys) from e

if __name__ == '__main__':
    mlflow.set_tracking_uri("http://127.0.0.1:5102")
    experiment_name = "credit_approval_pipeline"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    
    pipeline = Pipeline()
    pipeline.run()
