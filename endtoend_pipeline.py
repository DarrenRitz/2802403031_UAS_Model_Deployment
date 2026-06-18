import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score

os.makedirs("artifacts", exist_ok=True)

class DataCleaner:
    drop_cols = ["Unnamed: 0", "ID", "Customer_ID", "Name", "SSN", "Month"]

    valid_occupations = [
        "Teacher", "Doctor", "Accountant", "Writer", "Musician", "Engineer",
        "Architect", "Journalist", "Entrepreneur", "Developer", "Media_Manager",
        "Scientist", "Manager", "Lawyer", "Mechanic",
    ]

    valid_credit_mix = ["Standard", "Bad", "Good"]

    valid_payment_of_min = ["Yes", "No"]

    valid_payment_behaviour = [
        "Low_spent_Small_value_payments", "High_spent_Medium_value_payments",
        "Low_spent_Medium_value_payments", "High_spent_Small_value_payments",
        "Low_spent_Large_value_payments", "High_spent_Large_value_payments",
    ]

    loan_types = [
        "Student Loan", "Personal Loan", "Credit-Builder Loan",
        "Mortgage Loan", "Debt Consolidation Loan", "Payday Loan",
        "Auto Loan", "Home Equity Loan",
    ]

    outlier_cols = [
        "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
        "Num_Credit_Inquiries", "Num_of_Delayed_Payment", "Num_of_Loan",
    ]

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._drop_irrelevant(df)
        df = self._clean_age(df)
        df = self._clean_occupation(df)
        df = self._clean_annual_income(df)
        df = self._clean_monthly_salary(df)
        df = self._clean_numeric_non_negative(df)
        df = self._clean_num_of_loan(df)
        df = self._expand_loan_types(df)
        df = self._clean_delay(df)
        df = self._clean_extract_digits(df)
        df = self._clean_num_credit_inquiries(df)
        df = self._clean_credit_mix(df)
        df = self._clean_outstanding_debt(df)
        df = self._clean_credit_utilization(df)
        df = self._clean_credit_history_age(df)
        df = self._clean_payment_min(df)
        df = self._clean_emi(df)
        df = self._clean_amount_invested(df)
        df = self._clean_payment_behaviour(df)
        df = self._clean_monthly_balance(df)
        return df

    def _drop_irrelevant(self, df):
        cols_to_drop = [c for c in self.drop_cols if c in df.columns]
        return df.drop(columns=cols_to_drop)

    def _clean_age(self, df):
        df["Age"] = df["Age"].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
        df.loc[(df["Age"] < 17) | (df["Age"] > 100), "Age"] = np.nan
        return df

    def _clean_occupation(self, df):
        df["Occupation"] = df["Occupation"].apply(
            lambda x: x if x in self.valid_occupations else np.nan
        )
        return df

    def _clean_annual_income(self, df):
        df["Annual_Income"] = df["Annual_Income"].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
        df.loc[df["Annual_Income"] < 0, "Annual_Income"] = np.nan
        return df

    def _clean_monthly_salary(self, df):
        df["Monthly_Inhand_Salary"] = pd.to_numeric(df["Monthly_Inhand_Salary"], errors="coerce")
        df.loc[df["Monthly_Inhand_Salary"] < 0, "Monthly_Inhand_Salary"] = np.nan
        return df

    def _clean_numeric_non_negative(self, df):
        for col in ["Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df.loc[df[col] < 0, col] = np.nan
        return df

    def _clean_num_of_loan(self, df):
        df["Num_of_Loan"] = df["Num_of_Loan"].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
        df.loc[df["Num_of_Loan"] < 0, "Num_of_Loan"] = np.nan
        return df

    def _expand_loan_types(self, df):
        df["Type_of_Loan"] = df["Type_of_Loan"].astype(str).str.replace("and", ",", regex=False)
        df["Type_of_Loan"] = df["Type_of_Loan"].str.replace(r"\s+", " ", regex=True).str.strip()
        for loan_type in self.loan_types:
            col_name = loan_type.replace(" ", "_").lower()
            df[col_name] = df["Type_of_Loan"].apply(lambda x: 1 if loan_type in x else 0)
        df.drop(columns=["Type_of_Loan"], inplace=True)
        return df

    def _clean_delay(self, df):
        df["Delay_from_due_date"] = pd.to_numeric(df["Delay_from_due_date"], errors="coerce")
        df.loc[df["Delay_from_due_date"] < 0, "Delay_from_due_date"] = np.nan
        return df

    def _clean_extract_digits(self, df):
        for col in ["Num_of_Delayed_Payment", "Changed_Credit_Limit"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
                df.loc[df[col] < 0, col] = np.nan
        return df

    def _clean_num_credit_inquiries(self, df):
        df["Num_Credit_Inquiries"] = pd.to_numeric(df["Num_Credit_Inquiries"], errors="coerce")
        df.loc[df["Num_Credit_Inquiries"] < 0, "Num_Credit_Inquiries"] = np.nan
        return df

    def _clean_credit_mix(self, df):
        df["Credit_Mix"] = df["Credit_Mix"].apply(
            lambda x: x if x in self.valid_credit_mix else np.nan
        )
        return df

    def _clean_outstanding_debt(self, df):
        df["Outstanding_Debt"] = df["Outstanding_Debt"].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
        df.loc[df["Outstanding_Debt"] < 0, "Outstanding_Debt"] = np.nan
        return df

    def _clean_credit_utilization(self, df):
        df["Credit_Utilization_Ratio"] = pd.to_numeric(df["Credit_Utilization_Ratio"], errors="coerce")
        df.loc[df["Credit_Utilization_Ratio"] < 0, "Credit_Utilization_Ratio"] = np.nan
        return df

    def _clean_credit_history_age(self, df):
        df["Credit_History_Age"] = df["Credit_History_Age"].apply(
            lambda x:
            int(str(x).split()[0]) * 12
            if pd.notnull(x) and "year" in str(x).lower()
            else (
                int(str(x).split()[0])
                if pd.notnull(x)
                else np.nan
            )
        )
        return df

    def _clean_payment_min(self, df):
        df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].apply(
            lambda x: x if x in self.valid_payment_of_min else np.nan
        )
        return df

    def _clean_emi(self, df):
        df["Total_EMI_per_month"] = pd.to_numeric(df["Total_EMI_per_month"], errors="coerce")
        df.loc[df["Total_EMI_per_month"] < 0, "Total_EMI_per_month"] = np.nan
        return df

    def _clean_amount_invested(self, df):
        df["Amount_invested_monthly"] = df["Amount_invested_monthly"].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
        df.loc[df["Amount_invested_monthly"] < 0, "Amount_invested_monthly"] = np.nan
        return df

    def _clean_payment_behaviour(self, df):
        df["Payment_Behaviour"] = df["Payment_Behaviour"].apply(
            lambda x: x if x in self.valid_payment_behaviour else np.nan
        )
        return df

    def _clean_monthly_balance(self, df):
        if "Monthly_Balance" in df.columns:
            df["Monthly_Balance"] = df["Monthly_Balance"].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
            df.loc[df["Monthly_Balance"] < 0, "Monthly_Balance"] = np.nan
        return df

class Preprocessor:
    # Handles missing value imputation, outlier removal, scaling, and encoding.

    outlier_cols = [
        "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
        "Num_Credit_Inquiries", "Num_of_Delayed_Payment", "Num_of_Loan",
    ]

    credit_mix_map = {"Bad": 0, "Standard": 1, "Good": 2}

    def __init__(self):
        self._num_medians: dict = {}
        self._cat_modes: dict = {}
        self._scaler = RobustScaler()
        self._encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self._label_encoder = LabelEncoder()
        self._cat_cols_ohe: list = []
        self._num_cols: list = []

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series):
        X = X_train.copy()

        self._num_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
        self._cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

        for col in self._num_cols:
            self._num_medians[col] = X[col].median()
            X[col].fillna(self._num_medians[col], inplace=True)

        for col in self._cat_cols:
            mode_val = X[col].mode()
            self._cat_modes[col] = mode_val[0] if not mode_val.empty else "Unknown"
            X[col].fillna(self._cat_modes[col], inplace=True)

        for col in self.outlier_cols:
            if col in X.columns:
                X, y_train = self._drop_outliers_iqr(X, y_train, col)

        X[self._num_cols] = self._scaler.fit_transform(X[self._num_cols])

        # Ordinal encoding
        X["Credit_Mix"] = X["Credit_Mix"].map(self.credit_mix_map)

        # One Hot encoding
        self._cat_cols_ohe = [c for c in self._cat_cols if c != "Credit_Mix"]
        encoded = self._encoder.fit_transform(X[self._cat_cols_ohe])
        enc_df = pd.DataFrame(
            encoded,
            columns=self._encoder.get_feature_names_out(self._cat_cols_ohe),
            index=X.index,
        )
        X = pd.concat([X[self._num_cols + ["Credit_Mix"]], enc_df], axis=1)

        # Label encoding only target variable
        y_enc = self._label_encoder.fit_transform(y_train)

        return X, pd.Series(y_enc, index=y_train.index)

    def transform(self, X_test: pd.DataFrame, y_test: pd.Series = None):
        X = X_test.copy()

        for col in self._num_cols:
            if col in X.columns:
                X[col].fillna(self._num_medians.get(col, 0), inplace=True)

        for col in self._cat_cols:
            if col in X.columns:
                X[col].fillna(self._cat_modes.get(col, "Unknown"), inplace=True)

        num_present = [c for c in self._num_cols if c in X.columns]
        X[num_present] = self._scaler.transform(X[num_present])

        # Ordinal encoding
        X["Credit_Mix"] = X["Credit_Mix"].map(self.credit_mix_map)

        # One Hot encoding
        encoded = self._encoder.transform(X[self._cat_cols_ohe])
        enc_df = pd.DataFrame(
            encoded,
            columns=self._encoder.get_feature_names_out(self._cat_cols_ohe),
            index=X.index,
        )
        X = pd.concat([X[num_present + ["Credit_Mix"]], enc_df], axis=1)

        if y_test is not None:
            y_enc = self._label_encoder.transform(y_test)
            return X, pd.Series(y_enc, index=y_test.index)

        return X

    def get_label_encoder(self) -> LabelEncoder:
        return self._label_encoder

    @staticmethod
    def _drop_outliers_iqr(X, y, col):
        Q1, Q3 = X[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        mask = (X[col] >= Q1 - 1.5 * IQR) & (X[col] <= Q3 + 1.5 * IQR)
        return X[mask], y.loc[X[mask].index]

class ModelTrainer:
    param_grids = {
        "Random Forest": (
            RandomForestClassifier(random_state=8),
            {"n_estimators": [100, 200], "max_depth": [10, 20], "max_features": ["sqrt", "log2"]},
        ),
        "XGBoost": (
            XGBClassifier(random_state=8, use_label_encoder=False, eval_metric="mlogloss"),
            {"n_estimators": [100, 200], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=8),
            {"max_depth": [5, 10, 20], "min_samples_split": [2, 5, 10], "criterion": ["gini", "entropy"]},
        ),
    }

    def __init__(self, scoring: str = "f1_weighted"):
        self.scoring = scoring
        self.best_models: dict = {}
        self.run_ids: dict = {}

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        mlflow.set_experiment("Credit Score Classification")

        for name, (estimator, param_grid) in self.param_grids.items():
            print(f"Training {name}...")
            search = GridSearchCV(
                estimator, param_grid,
                cv=3, scoring="accuracy", n_jobs=-1, verbose=0
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            self.best_models[name] = best_model

            with mlflow.start_run(run_name=name) as run:
                mlflow.log_params(search.best_params_)
                y_pred = best_model.predict(X_train)
                mlflow.log_metric("train_accuracy", accuracy_score(y_train, y_pred))
                mlflow.log_metric("train_f1_weighted", f1_score(y_train, y_pred, average="weighted"))
                joblib.dump(best_model, f"artifacts/{name.replace(' ', '_').lower()}_model.pkl", compress=3)
                mlflow.sklearn.log_model(best_model, artifact_path="model")
                self.run_ids[name] = run.info.run_id
            print(f"Best params: {search.best_params_}")

        return self.best_models

    def get_run_ids(self) -> dict:
        return self.run_ids


def run_endtoend(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
    print("[EndToEnd] Cleaning data...")
    cleaner = DataCleaner()
    X_train_clean = cleaner.clean(X_train)
    X_test_clean  = cleaner.clean(X_test)

    print("[EndToEnd] Preprocessing...")
    preprocessor = Preprocessor()
    X_train_proc, y_train_enc = preprocessor.fit_transform(X_train_clean, y_train)
    X_test_proc,  y_test_enc  = preprocessor.transform(X_test_clean, y_test)
    joblib.dump(preprocessor, "artifacts/preprocessor.pkl")

    print("[EndToEnd] Training models...")
    trainer = ModelTrainer()
    trainer.train(X_train_proc, y_train_enc)

    return X_test_proc, y_test_enc, trainer.get_run_ids(), preprocessor