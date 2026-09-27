
# FLIPKART E-COMMERCE PROJECT
# MACHINE LEARNING MODEL


# 1. IMPORT LIBRARIES


import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)

warnings.filterwarnings("ignore")



# 2. CONFIGURATION


INPUT_FILE = os.path.join(
    # "processed_data",
    "feature_engineered_data",
    # "processed_data",
    "flipkart_feature_engineered.csv"
)

OUTPUT_FOLDER = "model_results"

MODEL_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "saved_models"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)



# 3. LOAD DATA


def load_data():

    print("\n" + "=" * 70)
    print("LOADING FEATURE-ENGINEERED DATA")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"""
Input file not found:

{INPUT_FILE}

Please run feature_engineering.py first.
"""
        )

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Rows    : {df.shape[0]:,}"
    )

    print(
        f"Columns : {df.shape[1]:,}"
    )

    return df



# 4. BASIC DATA INFORMATION


def display_dataset_information(df):

    print("\n" + "=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    for column in df.columns:
        print("-", column)

    print("\nMissing values:")
    print(
        df.isnull()
        .sum()
        .sort_values(ascending=False)
        .head(15)
    )



# 5. DEFINE TARGET VARIABLE

def define_target(df):

    print("\n" + "=" * 70)
    print("TARGET VARIABLE")
    print("=" * 70)

    target = "Price"

    if target not in df.columns:

        raise ValueError(
            "Target column 'Price' was not found."
        )

    print(
        f"Target variable: {target}"
    )

    print(
        "\nThe model will predict the selling price "
        "of a Flipkart product."
    )

    return target



# 6. REMOVE DATA LEAKAGE


def remove_leakage_features(df, target):

    print("\n" + "=" * 70)
    print("REMOVING DATA LEAKAGE")
    print("=" * 70)

    df = df.copy()

    # IMPORTANT:
  
    # These variables are directly calculated from Price.
    # Using them to predict Price would cause data leakage.

    leakage_columns = [
        # target,

        # Directly derived from Price
        "Price_Difference",
        "Price_Ratio",
        "Price_Per_Discount",
        "Log_Price",

        # Directly derived from Price
        "Estimated_Cost",
        "Estimated_Marketplace_Fee",
        "Estimated_Profit",
        "Estimated_Profit_Margin",
        "Profitable_Flag",

        # Interactions containing Price
        "Rating_Price_Interaction",
        "Price_Discount_Interaction",

        # Category/brand averages calculated using Price
        "Category_Avg_Price",
        "Price_vs_Category_Avg",
        "Brand_Avg_Price"
    ]

    removed = []

    for column in leakage_columns:

        if column in df.columns:

            df = df.drop(
                columns=column
            )

            removed.append(column)

    print("\nRemoved leakage columns:")

    for column in removed:

        print(
            f"- {column}"
        )

    # Confirm target still exists
    if target not in df.columns:

        raise ValueError(
            f"Target column '{target}' "
            "was accidentally removed."
        )

    print(
        f"\nTarget column retained: {target}"
    )

    print(
        f"\nRemaining columns: {df.shape[1]}"
    )

    return df



# 7. REMOVE UNNECESSARY COLUMNS

def remove_unnecessary_columns(df):

    print("\n" + "=" * 70)
    print("REMOVING UNNECESSARY COLUMNS")
    print("=" * 70)

    df = df.copy()

    unnecessary_columns = [
        "Product Name",
        "Image URLS"
    ]

    removed = []

    for column in unnecessary_columns:

        if column in df.columns:

            df = df.drop(
                columns=column
            )

            removed.append(column)

    print("\nRemoved columns:")

    for column in removed:

        print(
            f"- {column}"
        )

    return df


# 8. PREPARE X AND y

def prepare_features(df, target):

    print("\n" + "=" * 70)
    print("PREPARING FEATURES")
    print("=" * 70)

    X = df.drop(
        columns=[target],
        errors="ignore"
    )

    y = df[target]

    print(
        f"\nX shape: {X.shape}"
    )

    print(
        f"y shape: {y.shape}"
    )

    return X, y



# 9. IDENTIFY COLUMN TYPES

def identify_columns(X):

    print("\n" + "=" * 70)
    print("IDENTIFYING FEATURE TYPES")
    print("=" * 70)

    numerical_columns = (
        X.select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        X.select_dtypes(
            include=["object", "category", "string"]
        )
        .columns
        .tolist()
    )

    print(
        f"\nNumerical features   : "
        f"{len(numerical_columns)}"
    )

    print(
        f"Categorical features : "
        f"{len(categorical_columns)}"
    )

    print("\nNumerical columns:")

    for column in numerical_columns:

        print(
            f"- {column}"
        )

    print("\nCategorical columns:")

    for column in categorical_columns:

        print(
            f"- {column}"
        )

    return (
        numerical_columns,
        categorical_columns
    )



# 10. CREATE PREPROCESSING PIPELINE

def create_preprocessor(
    numerical_columns,
    categorical_columns
):

    print("\n" + "=" * 70)
    print("CREATING PREPROCESSING PIPELINE")
    print("=" * 70)


    # Numerical preprocessing

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    
    # Categorical preprocessing


    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )


    # Combine

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            ),

            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ],
        remainder="drop"
    )

    print(
        "Preprocessing pipeline created."
    )

    return preprocessor



# 11. TRAIN-TEST SPLIT

def split_data(X, y):

    print("\n" + "=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print(
        f"Training rows : {X_train.shape[0]:,}"
    )

    print(
        f"Testing rows  : {X_test.shape[0]:,}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )



# 12. DEFINE MODELS

def define_models():

    print("\n" + "=" * 70)
    print("DEFINING MACHINE LEARNING MODELS")
    print("=" * 70)

    models = {

        "Linear Regression":
            LinearRegression(),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            ),

        "Extra Trees":
            ExtraTreesRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
    }

    for model_name in models:

        print(
            f"- {model_name}"
        )

    return models



# 13. TRAIN MODELS

def train_models(
    models,
    preprocessor,
    X_train,
    y_train
):

    print("\n" + "=" * 70)
    print("TRAINING MODELS")
    print("=" * 70)

    trained_models = {}

    for name, model in models.items():

        print(
            f"\nTraining: {name}"
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "model",
                    model
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        trained_models[name] = pipeline

        print(
            f"{name} training completed."
        )

    return trained_models



# 14. EVALUATE MODELS


def evaluate_models(
    trained_models,
    X_test,
    y_test
):

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    results = []

    predictions = {}

    for name, model in trained_models.items():

        print(
            f"\nEvaluating: {name}"
        )

        y_pred = model.predict(
            X_test
        )

        predictions[name] = y_pred

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        mse = mean_squared_error(
            y_test,
            y_pred
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            y_test,
            y_pred
        )

        results.append({

            "Model": name,

            "MAE": round(
                mae,
                2
            ),

            "MSE": round(
                mse,
                2
            ),

            "RMSE": round(
                rmse,
                2
            ),

            "R2_Score": round(
                r2,
                4
            )
        })

    results_df = pd.DataFrame(
        results
    )

   
    # Sort by R² descending
    

    results_df = results_df.sort_values(
        by="R2_Score",
        ascending=False
    )

    print("\nModel Performance:")

    print(
        results_df.to_string(
            index=False
        )
    )

    return (
        results_df,
        predictions
    )



# 15. SAVE MODEL RESULTS


def save_results(results_df):

    output_file = os.path.join(
        OUTPUT_FOLDER,
        "model_comparison.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nModel comparison saved:\n"
        f"{output_file}"
    )



# 16. SAVE PREDICTIONS

def save_predictions(
    y_test,
    predictions
):

    prediction_df = pd.DataFrame({
        "Actual_Price":
            y_test.values
    })

    for name, prediction in predictions.items():

        safe_name = (
            name
            .replace(" ", "_")
            .lower()
        )

        prediction_df[
            f"{safe_name}_Prediction"
        ] = prediction

    output_file = os.path.join(
        OUTPUT_FOLDER,
        "model_predictions.csv"
    )

    prediction_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Predictions saved:\n"
        f"{output_file}"
    )



# 17. SELECT MODEL


def select_best_model(
    results_df,
    trained_models
):

    print("\n" + "=" * 70)
    print("SELECTING MODEL")
    print("=" * 70)

    # Highest R² is selected for this regression task
    best_model_name = (
        results_df
        .sort_values(
            "R2_Score",
            ascending=False
        )
        .iloc[0]["Model"]
    )

    best_model = trained_models[
        best_model_name
    ]

    print(
        f"\nSelected model based on highest "
        f"test R²: {best_model_name}"
    )

    return (
        best_model_name,
        best_model
    )



# 18. SAVE BEST MODEL


def save_best_model(
    best_model,
    best_model_name
):

    filename = (
        best_model_name
        .replace(" ", "_")
        .lower()
        + ".joblib"
    )

    model_file = os.path.join(
        MODEL_FOLDER,
        filename
    )

    joblib.dump(
        best_model,
        model_file
    )

    print(
        f"\nBest model saved:\n"
        f"{model_file}"
    )

    return model_file


# 19. ACTUAL VS PREDICTED PLOT


def plot_actual_vs_predicted(
    y_test,
    best_model,
    best_model_name,
    X_test
):

    print("\n" + "=" * 70)
    print("CREATING ACTUAL VS PREDICTED PLOT")
    print("=" * 70)

    y_pred = best_model.predict(
        X_test
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.6
    )

    # Perfect prediction line
    min_value = min(
        y_test.min(),
        y_pred.min()
    )

    max_value = max(
        y_test.max(),
        y_pred.max()
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    plt.xlabel(
        "Actual Price"
    )

    plt.ylabel(
        "Predicted Price"
    )

    plt.title(
        f"Actual vs Predicted Price - "
        f"{best_model_name}"
    )

    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_FOLDER,
        "actual_vs_predicted.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"Plot saved:\n{output_file}"
    )



# 20. RESIDUAL ANALYSIS


def plot_residuals(
    y_test,
    best_model,
    best_model_name,
    X_test
):

    print("\n" + "=" * 70)
    print("CREATING RESIDUAL PLOT")
    print("=" * 70)

    y_pred = best_model.predict(
        X_test
    )

    residuals = (
        y_test.values
        - y_pred
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.scatter(
        y_pred,
        residuals,
        alpha=0.6
    )

    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.xlabel(
        "Predicted Price"
    )

    plt.ylabel(
        "Residual"
    )

    plt.title(
        f"Residual Analysis - "
        f"{best_model_name}"
    )

    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_FOLDER,
        "residual_analysis.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"Residual plot saved:\n{output_file}"
    )



# 21. FEATURE IMPORTANCE


def get_feature_importance(
    best_model,
    best_model_name
):

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

 
    # Only tree-based models have feature_importances_
   

    model = best_model.named_steps[
        "model"
    ]

    if not hasattr(
        model,
        "feature_importances_"
    ):

        print(
            "\nFeature importance is not "
            "available for this model."
        )

        return None

    preprocessor = (
        best_model.named_steps[
            "preprocessor"
        ]
    )

    try:

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

    except Exception:

        print(
            "Could not retrieve feature names."
        )

        return None

    importance = (
        model.feature_importances_
    )

    importance_df = pd.DataFrame({

        "Feature":
            feature_names,

        "Importance":
            importance

    })

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

  
    # Save all feature importance
  

    output_file = os.path.join(
        OUTPUT_FOLDER,
        "feature_importance.csv"
    )

    importance_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nFeature importance saved:\n"
        f"{output_file}"
    )

    print(
        "\nTop 20 important features:"
    )

    print(
        importance_df
        .head(20)
        .to_string(index=False)
    )

    
    # Plot top 15
   

    top_features = (
        importance_df
        .head(15)
        .sort_values(
            "Importance"
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"Top 15 Feature Importance - "
        f"{best_model_name}"
    )

    plt.tight_layout()

    plot_file = os.path.join(
        OUTPUT_FOLDER,
        "feature_importance.png"
    )

    plt.savefig(
        plot_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"\nFeature importance plot saved:\n"
        f"{plot_file}"
    )

    return importance_df


# 22. SAMPLE PREDICTION


def sample_prediction(
    model,
    X_test,
    y_test
):

    print("\n" + "=" * 70)
    print("SAMPLE PRICE PREDICTIONS")
    print("=" * 70)

    sample_size = min(
        10,
        len(X_test)
    )

    sample_X = X_test.iloc[
        :sample_size
    ]

    sample_y = y_test.iloc[
        :sample_size
    ]

    predictions = model.predict(
        sample_X
    )

    prediction_table = pd.DataFrame({

        "Actual_Price":
            sample_y.values,

        "Predicted_Price":
            predictions,

        "Difference":
            sample_y.values
            - predictions

    })

    prediction_table[
        "Absolute_Error"
    ] = (
        prediction_table[
            "Difference"
        ].abs()
    )

    print(
        prediction_table
        .round(2)
        .to_string(index=False)
    )

    output_file = os.path.join(
        OUTPUT_FOLDER,
        "sample_predictions.csv"
    )

    prediction_table.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nSample predictions saved:\n"
        f"{output_file}"
    )



# 23. PREDICTION FUNCTION


def predict_price(
    model,
    product_data
):

    """
    Predict selling price for a new product.

    product_data must be a pandas DataFrame
    containing the same input features used during training.
    """

    prediction = model.predict(
        product_data
    )

    return prediction



# 24. MAIN FUNCTION


def main():

    print("\n")

    print("=" * 70)

    print(
        "FLIPKART MACHINE LEARNING PIPELINE"
    )

    print("=" * 70)

   
    # STEP 1
    # Load data
   

    df = load_data()

   
    # STEP 2
    # Dataset information
   

    display_dataset_information(
        df
    )

  
    # STEP 3
    # Define target
   

    target = define_target(
        df
    )

   
    # STEP 4
    # Remove data leakage
   

    df = remove_leakage_features(
        df,
        target
    )

   
    # STEP 5
    # Remove unnecessary columns
    

    df = remove_unnecessary_columns(
        df
    )

   
    # STEP 6
    # Prepare X and y
   

    X, y = prepare_features(
        df,
        target
    )

    
    # STEP 7
    # Identify feature types
   

    (
        numerical_columns,
        categorical_columns
    ) = identify_columns(X)

   
    # STEP 8
    # Create preprocessing
   

    preprocessor = create_preprocessor(
        numerical_columns,
        categorical_columns
    )

  
    # STEP 9
    # Train/test split
   

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(
        X,
        y
    )

   
    # STEP 10
    # Define models
 

    models = define_models()

   
    # STEP 11
    # Train models
   

    trained_models = train_models(
        models,
        preprocessor,
        X_train,
        y_train
    )

   
    # STEP 12
    # Evaluate models


    (
        results_df,
        predictions
    ) = evaluate_models(
        trained_models,
        X_test,
        y_test
    )

   
    # STEP 13
    # Save comparison
   

    save_results(
        results_df
    )

    
    # STEP 14
    # Save predictions
   

    save_predictions(
        y_test,
        predictions
    )

  
    # STEP 15
    # Select model
   

    (
        best_model_name,
        best_model
    ) = select_best_model(
        results_df,
        trained_models
    )


    # STEP 16
    # Save model
    

    save_best_model(
        best_model,
        best_model_name
    )

  
    # STEP 17
    # Actual vs predicted
   

    plot_actual_vs_predicted(
        y_test,
        best_model,
        best_model_name,
        X_test
    )

    
    # STEP 18
    # Residual analysis
    

    plot_residuals(
        y_test,
        best_model,
        best_model_name,
        X_test
    )

  
    # STEP 19
    # Feature importance
    

    get_feature_importance(
        best_model,
        best_model_name
    )

    
    # STEP 20
    # Sample predictions
   

    sample_prediction(
        best_model,
        X_test,
        y_test
    )

   
    # FINAL MESSAGE


    print("\n")

    print("=" * 70)

    print(
        "MACHINE LEARNING COMPLETED SUCCESSFULLY"
    )

    print("=" * 70)

    print(
        f"\nSelected model: "
        f"{best_model_name}"
    )

    print(
        "\nOutput folder:"
    )

    print(
        f"{OUTPUT_FOLDER}/"
    )

    print(
        "\nGenerated files:"
    )

    print(
        "1. model_comparison.csv"
    )

    print(
        "2. model_predictions.csv"
    )

    print(
        "3. sample_predictions.csv"
    )

    print(
        "4. feature_importance.csv"
    )

    print(
        "5. actual_vs_predicted.png"
    )

    print(
        "6. residual_analysis.png"
    )

    print(
        "7. feature_importance.png"
    )

    print(
        "8. saved_models/"
    )

    print("\nNext step:")
    print(
        "Use the saved model in Streamlit "
        "for interactive price prediction."
    )



# 25. RUN


if __name__ == "__main__":

    main()