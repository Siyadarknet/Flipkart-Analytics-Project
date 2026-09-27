
# FLIPKART E-COMMERCE PROJECT
# FEATURE ENGINEERING


# 1. IMPORT LIBRARIES


import os
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    MinMaxScaler
)

warnings.filterwarnings("ignore")



# 2. CONFIGURATION


INPUT_FILE = os.path.join(
    "processed_data",
    "flipkart_cleaned.csv"
)

OUTPUT_FOLDER = "feature_engineered_data"

OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "flipkart_feature_engineered.csv"
)

ML_FILE = os.path.join(
    OUTPUT_FOLDER,
    "flipkart_ml_ready.csv"
)


# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)



# 3. LOAD DATA


def load_data():

    print("\n" + "=" * 60)
    print("LOADING CLEANED DATA")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"\nInput file not found:\n{INPUT_FILE}\n\n"
            "Please run data_cleaning.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Dataset loaded successfully.")
    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]:,}")

    return df



# 4. BASIC DATA TYPE CONVERSION


def convert_data_types(df):

    print("\n" + "=" * 60)
    print("CONVERTING DATA TYPES")
    print("=" * 60)

    # Numeric columns
    numeric_columns = [
        "Price",
        "Original Prices",
        "Discount rates",
        "Discount_Amount",
        "Discount_Rate_Calculated",
        "Ratings"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Text columns
    text_columns = [
        "Product Name",
        "Category",
        "Brand",
        "Brand_Normalized"
    ]

    for column in text_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
            )

    print("Data types converted.")

    return df



# 5. HANDLE MISSING VALUES


def handle_missing_values(df):

    print("\n" + "=" * 60)
    print("HANDLING MISSING VALUES")
    print("=" * 60)

    
    # Numeric columns
    

    numeric_columns = [
        "Price",
        "Original Prices",
        "Discount rates",
        "Discount_Amount",
        "Discount_Rate_Calculated",
        "Ratings"
    ]

    for column in numeric_columns:

        if column in df.columns:

            median_value = df[column].median()

            if pd.notna(median_value):

                df[column] = df[column].fillna(
                    median_value
                )

    
    # Text columns
   

    text_columns = [
        "Product Name",
        "Category",
        "Brand",
        "Brand_Normalized"
    ]

    for column in text_columns:

        if column in df.columns:

            df[column] = df[column].fillna(
                "Unknown"
            )

    print("Missing values handled.")

    return df



# 6. CREATE PRICE FEATURES


def create_price_features(df):

    print("\n" + "=" * 60)
    print("CREATING PRICE FEATURES")
    print("=" * 60)

   
    # Price difference
   

    if (
        "Original Prices" in df.columns
        and "Price" in df.columns
    ):

        df["Price_Difference"] = (
            df["Original Prices"] - df["Price"]
        )

  
    # Price ratio
    

    if (
        "Original Prices" in df.columns
        and "Price" in df.columns
    ):

        df["Price_Ratio"] = np.where(
            df["Original Prices"] > 0,
            df["Price"] / df["Original Prices"],
            1
        )

    
    # Price per percentage discount
   

    if (
        "Price" in df.columns
        and "Discount rates" in df.columns
    ):

        df["Price_Per_Discount"] = np.where(
            df["Discount rates"] > 0,
            df["Price"] / df["Discount rates"],
            df["Price"]
        )

   
    # Log transformed price
    # Useful for ML models
   

    if "Price" in df.columns:

        df["Log_Price"] = np.log1p(
            df["Price"].clip(lower=0)
        )

   
    # Price bands
  

    if "Price" in df.columns:

        bins = [
            -np.inf,
            500,
            1000,
            5000,
            10000,
            25000,
            50000,
            100000,
            np.inf
        ]

        labels = [
            "Under_500",
            "500_1000",
            "1000_5000",
            "5000_10000",
            "10000_25000",
            "25000_50000",
            "50000_100000",
            "Above_100000"
        ]

        df["Price_Band"] = pd.cut(
            df["Price"],
            bins=bins,
            labels=labels,
            include_lowest=True
        )

    print("Price features created.")

    return df



# 7. CREATE DISCOUNT FEATURES


def create_discount_features(df):

    print("\n" + "=" * 60)
    print("CREATING DISCOUNT FEATURES")
    print("=" * 60)

  
    # Effective discount amount
   

    if (
        "Original Prices" in df.columns
        and "Price" in df.columns
    ):

        df["Discount_Amount"] = (
            df["Original Prices"] - df["Price"]
        )

   
    # Effective discount percentage
    

    if (
        "Original Prices" in df.columns
        and "Price" in df.columns
    ):

        df["Effective_Discount"] = np.where(
            df["Original Prices"] > 0,
            (
                (
                    df["Original Prices"]
                    - df["Price"]
                )
                / df["Original Prices"]
            ) * 100,
            0
        )

   
    # Discount bands
  

    if "Discount rates" in df.columns:

        bins = [
            -np.inf,
            0,
            10,
            20,
            30,
            50,
            70,
            100,
            np.inf
        ]

        labels = [
            "No_Discount",
            "1_10",
            "11_20",
            "21_30",
            "31_50",
            "51_70",
            "71_100",
            "Above_100"
        ]

        df["Discount_Band"] = pd.cut(
            df["Discount rates"],
            bins=bins,
            labels=labels,
            include_lowest=True
        )

   
    # High discount flag
   

    if "Discount rates" in df.columns:

        df["High_Discount_Flag"] = np.where(
            df["Discount rates"] >= 30,
            1,
            0
        )

   
    # Very high discount flag
    

    if "Discount rates" in df.columns:

        df["Very_High_Discount_Flag"] = np.where(
            df["Discount rates"] >= 50,
            1,
            0
        )

    print("Discount features created.")

    return df



# 8. CREATE RATING FEATURES


def create_rating_features(df):

    print("\n" + "=" * 60)
    print("CREATING RATING FEATURES")
    print("=" * 60)

    if "Ratings" not in df.columns:

        print("Ratings column not available.")

        return df

    
    # Rating bands
    

    bins = [
        -np.inf,
        2,
        3,
        3.5,
        4,
        4.5,
        5
    ]

    labels = [
        "Very_Low",
        "Low",
        "Average",
        "Good",
        "Very_Good",
        "Excellent"
    ]

    df["Rating_Band"] = pd.cut(
        df["Ratings"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    
    # Rating flag
    

    df["High_Rating_Flag"] = np.where(
        df["Ratings"] >= 4,
        1,
        0
    )

    
    # Excellent rating flag
   

    df["Excellent_Rating_Flag"] = np.where(
        df["Ratings"] >= 4.5,
        1,
        0
    )

   
    # Rating squared
  

    df["Rating_Squared"] = (
        df["Ratings"] ** 2
    )

    print("Rating features created.")

    return df



# 9. CREATE CATEGORY FEATURES


def create_category_features(df):

    print("\n" + "=" * 60)
    print("CREATING CATEGORY FEATURES")
    print("=" * 60)

    if "Category" not in df.columns:

        return df

    
    # Category frequency
   

    category_frequency = (
        df["Category"]
        .value_counts()
    )

    df["Category_Product_Count"] = (
        df["Category"]
        .map(category_frequency)
    )

    
    # Category average price
   

    if "Price" in df.columns:

        category_avg_price = (
            df.groupby("Category")["Price"]
            .mean()
        )

        df["Category_Avg_Price"] = (
            df["Category"]
            .map(category_avg_price)
        )

    
    # Difference from category average price
   

    if (
        "Price" in df.columns
        and "Category_Avg_Price" in df.columns
    ):

        df["Price_vs_Category_Avg"] = (
            df["Price"]
            - df["Category_Avg_Price"]
        )

    print("Category features created.")

    return df



# 10. CREATE BRAND FEATURES


def create_brand_features(df):

    print("\n" + "=" * 60)
    print("CREATING BRAND FEATURES")
    print("=" * 60)

    if "Brand_Normalized" not in df.columns:

        if "Brand" in df.columns:

            df["Brand_Normalized"] = (
                df["Brand"]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
                .str.lower()
            )

        else:

            return df

    
    # Brand product count
    

    brand_frequency = (
        df["Brand_Normalized"]
        .value_counts()
    )

    df["Brand_Product_Count"] = (
        df["Brand_Normalized"]
        .map(brand_frequency)
    )

    
    # Brand average price
    

    if "Price" in df.columns:

        brand_avg_price = (
            df.groupby("Brand_Normalized")["Price"]
            .mean()
        )

        df["Brand_Avg_Price"] = (
            df["Brand_Normalized"]
            .map(brand_avg_price)
        )

   
    # Brand average discount
   

    if "Discount rates" in df.columns:

        brand_avg_discount = (
            df.groupby("Brand_Normalized")[
                "Discount rates"
            ].mean()
        )

        df["Brand_Avg_Discount"] = (
            df["Brand_Normalized"]
            .map(brand_avg_discount)
        )

    print("Brand features created.")

    return df



# 11. CREATE PRODUCT NAME FEATURES


def create_product_features(df):

    print("\n" + "=" * 60)
    print("CREATING PRODUCT NAME FEATURES")
    print("=" * 60)

    if "Product Name" not in df.columns:

        return df

   
    # Product name length
    

    df["Product_Name_Length"] = (
        df["Product Name"]
        .astype(str)
        .str.len()
    )

    
    # Number of words
   

    df["Product_Name_Word_Count"] = (
        df["Product Name"]
        .astype(str)
        .str.split()
        .str.len()
    )

    
    # Number of digits
   

    df["Product_Name_Digit_Count"] = (
        df["Product Name"]
        .astype(str)
        .str.count(r"\d")
    )

    
    # Number of uppercase characters
   

    df["Product_Name_Upper_Count"] = (
        df["Product Name"]
        .astype(str)
        .str.count(r"[A-Z]")
    )

    print("Product name features created.")

    return df


# 12. CREATE PROFITABILITY FEATURES


def create_profit_features(df):

    print("\n" + "=" * 60)
    print("CREATING PROFITABILITY FEATURES")
    print("=" * 60)

    if "Price" not in df.columns:

        return df

    
    # IMPORTANT:
    # Dataset does NOT contain actual seller cost.
  
    # These are illustrative assumptions only.
    
    COST_RATE = 0.60
    MARKETPLACE_FEE_RATE = 0.12
    SHIPPING_COST = 40

    # Estimated product cost
    df["Estimated_Cost"] = (
        df["Price"] * COST_RATE
    )

    # Estimated marketplace/payment fee
    df["Estimated_Marketplace_Fee"] = (
        df["Price"]
        * MARKETPLACE_FEE_RATE
    )

    # Estimated profit
    df["Estimated_Profit"] = (
        df["Price"]
        - df["Estimated_Cost"]
        - df["Estimated_Marketplace_Fee"]
        - SHIPPING_COST
    )

    # Profit margin
    df["Estimated_Profit_Margin"] = np.where(
        df["Price"] > 0,
        (
            df["Estimated_Profit"]
            / df["Price"]
        ) * 100,
        0
    )

    # Profit flag
    df["Profitable_Flag"] = np.where(
        df["Estimated_Profit"] > 0,
        1,
        0
    )

    print(
        "Illustrative profitability features created."
    )

    return df


# 13. CREATE COMBINED FEATURES

def create_combined_features(df):

    print("\n" + "=" * 60)
    print("CREATING COMBINED FEATURES")
    print("=" * 60)

    
    # Price × Discount
   
    if (
        "Price" in df.columns
        and "Discount rates" in df.columns
    ):

        df["Price_Discount_Interaction"] = (
            df["Price"]
            * df["Discount rates"]
        )


    # Rating × Discount
    

    if (
        "Ratings" in df.columns
        and "Discount rates" in df.columns
    ):

        df["Rating_Discount_Interaction"] = (
            df["Ratings"]
            * df["Discount rates"]
        )

    
    # Rating × Price
    

    if (
        "Ratings" in df.columns
        and "Price" in df.columns
    ):

        df["Rating_Price_Interaction"] = (
            df["Ratings"]
            * df["Price"]
        )

   
    # Value Score
    
    # Higher rating + higher discount
   

    if (
        "Ratings" in df.columns
        and "Discount rates" in df.columns
    ):

        df["Value_Score"] = (
            df["Ratings"]
            * (
                1
                + df["Discount rates"] / 100
            )
        )

    print("Combined features created.")

    return df



# 14. REMOVE EXTREME INVALID VALUES


def clean_engineered_values(df):

    print("\n" + "=" * 60)
    print("CLEANING ENGINEERED FEATURES")
    print("=" * 60)

    # Replace infinite values
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Numeric columns
    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        median_value = df[column].median()

        if pd.notna(median_value):

            df[column] = df[column].fillna(
                median_value
            )

    print("Engineered values cleaned.")

    return df



# 15. ENCODE CATEGORICAL FEATURES


def encode_categorical_features(df):

    print("\n" + "=" * 60)
    print("ENCODING CATEGORICAL FEATURES")
    print("=" * 60)

    categorical_columns = [
        "Category",
        "Brand_Normalized",
        "Price_Band",
        "Discount_Band",
        "Rating_Band"
    ]

    existing_columns = [
        column
        for column in categorical_columns
        if column in df.columns
    ]

  
    # Label Encoding
  

    for column in existing_columns:

        encoder = LabelEncoder()

        values = (
            df[column]
            .astype(str)
            .fillna("Unknown")
        )

        df[f"{column}_Encoded"] = (
            encoder.fit_transform(values)
        )

    print(
        f"Encoded columns: {existing_columns}"
    )

    return df



# 16. CREATE ONE-HOT ENCODED DATASET


def create_one_hot_dataset(df):

   
    print("CREATING ONE-HOT ENCODED DATASET")
   

    ml_df = df.copy()

    # Remove high-cardinality text columns
    columns_to_remove = [
        "Product Name",
        "Image URLS"
    ]

    for column in columns_to_remove:

        if column in ml_df.columns:

            ml_df = ml_df.drop(
                columns=column
            )

    # Convert categorical columns
    categorical_columns = [
        "Category",
        "Brand_Normalized",
        "Price_Band",
        "Discount_Band",
        "Rating_Band"
    ]

    categorical_columns = [
        column
        for column in categorical_columns
        if column in ml_df.columns
    ]

    # One-hot encoding
    ml_df = pd.get_dummies(
        ml_df,
        columns=categorical_columns,
        prefix=categorical_columns,
        dtype=int
    )

    print(
        f"ML-ready dataset shape: {ml_df.shape}"
    )

    return ml_df



# 17. SCALE NUMERICAL FEATURES


def scale_numeric_features(df):

  
    print("SCALING NUMERICAL FEATURES")
  

    scaled_df = df.copy()

    numeric_columns = scaled_df.select_dtypes(
        include=np.number
    ).columns.tolist()

    # Do not scale target-like/business columns
    exclude_columns = [
        "Price",
        "Original Prices",
        "Ratings"
    ]

    scale_columns = [
        column
        for column in numeric_columns
        if column not in exclude_columns
    ]

    if len(scale_columns) == 0:

        return scaled_df

    scaler = StandardScaler()

    scaled_values = scaler.fit_transform(
        scaled_df[scale_columns]
    )

    scaled_df[
        [f"{column}_Scaled"
         for column in scale_columns]
    ] = scaled_values

    print(
        f"Scaled {len(scale_columns)} numerical features."
    )

    return scaled_df



# 18. FEATURE IMPORTANCE-READY DATA


def create_feature_summary(df):

   
    print("CREATING FEATURE SUMMARY")
   

    summary = pd.DataFrame({
        "Feature": df.columns,
        "Data_Type": [
            str(df[column].dtype)
            for column in df.columns
        ],
        "Missing_Values": [
            df[column].isna().sum()
            for column in df.columns
        ],
        "Unique_Values": [
            df[column].nunique()
            for column in df.columns
        ]
    })

    summary["Missing_Percentage"] = (
        summary["Missing_Values"]
        / len(df)
        * 100
    )

    summary = summary.sort_values(
        by="Missing_Percentage",
        ascending=False
    )

    output_file = os.path.join(
        OUTPUT_FOLDER,
        "feature_summary.csv"
    )

    summary.to_csv(
        output_file,
        index=False
    )

    print(
        f"Feature summary saved:\n{output_file}"
    )

    return summary



# 19. SAVE FEATURE-ENGINEERED DATA


def save_feature_engineered_data(df):

   
    print("SAVING FEATURE-ENGINEERED DATA")
   

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Feature-engineered dataset saved:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"Rows    : {df.shape[0]:,}"
    )

    print(
        f"Columns : {df.shape[1]:,}"
    )


# 20. SAVE ML-READY DATA


def save_ml_ready_data(df):

  
    print("CREATING ML-READY DATASET")
   

    ml_df = create_one_hot_dataset(df)

   
    # Fill remaining missing values
   

    numeric_columns = ml_df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        median_value = ml_df[column].median()

        if pd.notna(median_value):

            ml_df[column] = (
                ml_df[column]
                .fillna(median_value)
            )

   
    # Remove duplicate columns if any
  

    ml_df = ml_df.loc[
        :,
        ~ml_df.columns.duplicated()
    ]


    # Save
    

    ml_df.to_csv(
        ML_FILE,
        index=False
    )

    print(
        f"ML-ready dataset saved:\n"
        f"{ML_FILE}"
    )

    print(
        f"Rows    : {ml_df.shape[0]:,}"
    )

    print(
        f"Columns : {ml_df.shape[1]:,}"
    )

    return ml_df



# 21. DISPLAY FEATURE INFORMATION


def display_feature_information(df):

   
    print("FEATURE ENGINEERING SUMMARY")
   

    print(
        f"\nOriginal/processed rows : {df.shape[0]:,}"
    )

    print(
        f"Final feature count     : {df.shape[1]:,}"
    )

    print("\nFeature names:")

    for i, column in enumerate(
        df.columns,
        start=1
    ):

        print(
            f"{i:3}. {column}"
        )



# 22. MAIN FUNCTION

def main():

    print("\n")
    
    print("FLIPKART FEATURE ENGINEERING PIPELINE")
  

    
    # Step 1: Load
    

    df = load_data()

    
    # Step 2: Convert data types
    

    df = convert_data_types(df)

    
    # Step 3: Handle missing values
    

    df = handle_missing_values(df)

    
    # Step 4: Price features
    

    df = create_price_features(df)

    
    # Step 5: Discount features
    

    df = create_discount_features(df)

    
    # Step 6: Rating features
    

    df = create_rating_features(df)

    
    # Step 7: Category features
    

    df = create_category_features(df)

    
    # Step 8: Brand features
    

    df = create_brand_features(df)

    
    # Step 9: Product features
    

    df = create_product_features(df)

    
    # Step 10: Profit features
    

    df = create_profit_features(df)

    
    # Step 11: Combined features
    

    df = create_combined_features(df)

    
    # Step 12: Clean engineered values
    

    df = clean_engineered_values(df)

    
    # Step 13: Encode categorical features
    

    df = encode_categorical_features(df)

    
    # Step 14: Save feature-engineered dataset
    

    save_feature_engineered_data(df)

    
    # Step 15: Feature summary
    

    create_feature_summary(df)

    
    # Step 16: ML-ready dataset
    

    ml_df = save_ml_ready_data(df)

    
    # Step 17: Display information
    

    display_feature_information(df)

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nGenerated files:")

    print(
        f"1. {OUTPUT_FILE}"
    )

    print(
        f"2. {ML_FILE}"
    )

    print(
        f"3. {OUTPUT_FOLDER}/feature_summary.csv"
    )

    print("\nNext step:")
    print(
        "Use the ML-ready dataset for model building."
    )


# 23. RUN PROGRAM


if __name__ == "__main__":
    main()