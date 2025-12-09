import pandas as pd

pd.set_option("display.max_columns", None)

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


class DataLoader:
    def __init__(self):
        self.data = None

    def load_dataset(self, path="data/healthcare-dataset-stroke-data.csv"):
        self.data = pd.read_csv(path)
        return self.data

    def preprocess_data(self):
        categorical_cols = [
            "gender",
            "ever_married",
            "work_type",
            "Residence_type",
            "smoking_status",
        ]

        # One-hot encode categorical columns
        encode = pd.get_dummies(self.data[categorical_cols], prefix=categorical_cols)
        self.data = pd.concat([encode, self.data], axis=1)
        self.data.drop(columns=categorical_cols, inplace=True, axis=1)

        # Fill NaNs only in 'bmi' column
        self.data["bmi"] = self.data["bmi"].fillna(0)

        # Drop 'id' column
        self.data.drop(["id"], axis=1, inplace=True)

        # Convert any remaining 'True'/'False' strings to integers 1/0
        self._convert_boolean_strings()

    def _convert_boolean_strings(self):
        # Find columns with object type that might have 'True'/'False' as strings
        obj_cols = self.data.select_dtypes(include=["object"]).columns
        for col in obj_cols:
            if self.data[col].isin(["True", "False"]).all():
                self.data[col] = self.data[col].map({"True": 1, "False": 0}).astype(int)

    def split_data(self, test_size=0.2, random_state=42):
        X = self.data.drop("stroke", axis=1)
        y = self.data["stroke"]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def oversample_data(self, X_train, y_train):
        oversample = RandomOverSampler(sampling_strategy="minority")
        x_np = X_train.to_numpy()
        y_np = y_train.to_numpy()
        X_over, y_over = oversample.fit_resample(x_np, y_np)
        X_over_df = pd.DataFrame(X_over, columns=X_train.columns)
        y_over_df = pd.Series(y_over, name=y_train.name)

        # Convert boolean strings after oversampling
        X_over_df = self._convert_boolean_strings_df(X_over_df)

        return X_over_df, y_over_df
