
# Flipkart Dataset - Data Cleaning

import os
import re
import zipfile
import numpy as np
import pandas as pd

# 1. CONFIGURATION

ZIP_FILE = r"archive (3)(1).zip"

EXTRACT_FOLDER = "flipkart_data"

OUTPUT_FOLDER = "processed_data"

OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "flipkart_cleaned.csv"
)



# 2. EXTRACT ZIP FILE

def extract_dataset(zip_file, extract_folder):

    if not os.path.exists(zip_file):
        raise FileNotFoundError(
            f"ZIP file not found: {zip_file}"
        )

    if not os.path.exists(extract_folder):

        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(extract_folder)

        print("Dataset extracted successfully.")

    else:
        print("Dataset folder already exists.")



# 3. FIND CSV FILES

def find_csv_files(folder):

    csv_files = []

    for root, dirs, files in os.walk(folder):

        for file in files:

            if file.lower().endswith(".csv"):

                csv_files.append(
                    os.path.join(root, file)
                )

    if len(csv_files) == 0:
        raise FileNotFoundError(
            "No CSV files found."
        )

    return sorted(csv_files)



# 4. GET CATEGORY NAME


def get_category_name(file_path):

    file_name = os.path.basename(file_path)

    category = os.path.splitext(file_name)[0]

    return category.lower().strip()


# 5. CLEAN MONEY COLUMNS


def clean_money(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    # Remove currency symbols and commas
    value = re.sub(r"[₹$€£,\s]", "", value)

    # Keep numbers and decimal point
    value = re.sub(r"[^0-9.]", "", value)

    if value == "":
        return np.nan

    try:
        return float(value)

    except ValueError:
        return np.nan



# 6. CLEAN DISCOUNT


def clean_discount(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    # Remove % and words such as "off"
    value = value.replace("%", "")
    value = value.replace("off", "")
    value = value.strip()

    if value in ["", "nan", "none", "null", "-"]:
        return np.nan

    # Extract numeric value
    match = re.search(
        r"\d+(?:\.\d+)?",
        value
    )

    if match:

        try:
            return float(match.group())

        except ValueError:
            return np.nan

    return np.nan



# 7. CLEAN TEXT


def clean_text(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value.lower() in [
        "",
        "nan",
        "none",
        "null",
        "n/a",
        "na"
    ]:
        return np.nan

    return value



# 8. LOAD ONE CSV


def load_and_clean_csv(file_path):

    print(
        f"\nLoading: {os.path.basename(file_path)}"
    )

    df = pd.read_csv(file_path)

  
    # Add Category
   

    df["Category"] = get_category_name(
        file_path
    )

  
    # Remove Image URLs


    if "Image URLS" in df.columns:

        df = df.drop(
            columns=["Image URLS"]
        )


    # Clean Product Name
    

    if "Product Name" in df.columns:

        df["Product Name"] = (
            df["Product Name"]
            .apply(clean_text)
        )

    # Clean Brand


    if "Brand" in df.columns:

        df["Brand"] = (
            df["Brand"]
            .apply(clean_text)
        )

        # Normalized brand
        df["Brand_Normalized"] = (
            df["Brand"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

  
    # Clean Price
  

    if "Price" in df.columns:

        df["Price"] = (
            df["Price"]
            .apply(clean_money)
        )

    # Clean Original Price
   

    if "Original Prices" in df.columns:

        df["Original Prices"] = (
            df["Original Prices"]
            .apply(clean_money)
        )


    # Clean Discount
  

    if "Discount rates" in df.columns:

        df["Discount rates"] = (
            df["Discount rates"]
            .apply(clean_discount)
        )

    # --------------------------------------------------------
    # Clean Ratings
    # --------------------------------------------------------

    if "Ratings" in df.columns:

        df["Ratings"] = (
            pd.to_numeric(
                df["Ratings"],
                errors="coerce"
            )
        )

        # Keep ratings within valid range
        df.loc[
            (df["Ratings"] < 0) |
            (df["Ratings"] > 5),
            "Ratings"
        ] = np.nan

  
    # Fill missing Original Price
  

    if (
        "Original Prices" in df.columns
        and "Price" in df.columns
    ):

        df["Original Prices"] = (
            df["Original Prices"]
            .fillna(df["Price"])
        )

 
    # Fill missing Discount


    if "Discount rates" in df.columns:

        df["Discount rates"] = (
            df["Discount rates"]
            .fillna(0)
        )


    # Calculate Discount Amount


    if (
        "Original Prices" in df.columns
        and "Price" in df.columns
    ):

        df["Discount_Amount"] = (
            df["Original Prices"]
            - df["Price"]
        )


    # Recalculate Discount %


    if (
        "Original Prices" in df.columns
        and "Price" in df.columns
    ):

        calculated_discount = np.where(
            df["Original Prices"] > 0,

            (
                (
                    df["Original Prices"]
                    - df["Price"]
                )
                /
                df["Original Prices"]
            ) * 100,

            0
        )

        # Use calculated discount
        # when source discount is missing
        df["Discount rates"] = np.where(
            df["Discount rates"].isna(),
            calculated_discount,
            df["Discount rates"]
        )

    # Remove duplicate rows
   

    df = df.drop_duplicates()

    return df



# 9. LOAD ALL DATASETS


def load_all_datasets(csv_files):

    datasets = []

    for file_path in csv_files:

        df = load_and_clean_csv(
            file_path
        )

        datasets.append(df)

        print(
            f"Rows after cleaning: {len(df)}"
        )

    return datasets



# 10. STANDARDIZE COLUMNS


def standardize_columns(datasets):

    all_columns = set()

    for df in datasets:
        all_columns.update(
            df.columns
        )

    all_columns = list(all_columns)

    standardized = []

    for df in datasets:

        for column in all_columns:

            if column not in df.columns:

                df[column] = np.nan

        df = df[all_columns]

        standardized.append(df)

    return standardized



# 11. COMBINE ALL DATASETS


def combine_datasets(datasets):

    raw = pd.concat(
        datasets,
        ignore_index=True
    )

    return raw



# 12. FINAL CLEANING


def final_cleaning(df):

    # Remove completely empty rows
    df = df.dropna(
        how="all"
    )

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove products without product name
    if "Product Name" in df.columns:

        df = df[
            df["Product Name"].notna()
        ]

    # Price cannot be negative
    if "Price" in df.columns:

        df.loc[
            df["Price"] < 0,
            "Price"
        ] = np.nan

    # Original price cannot be negative
    if "Original Prices" in df.columns:

        df.loc[
            df["Original Prices"] < 0,
            "Original Prices"
        ] = np.nan

    # Make sure original price >= selling price
    if (
        "Original Prices" in df.columns
        and "Price" in df.columns
    ):

        df["Original Prices"] = np.maximum(
            df["Original Prices"],
            df["Price"]
        )

    # Recalculate discount amount
    if (
        "Original Prices" in df.columns
        and "Price" in df.columns
    ):

        df["Discount_Amount"] = (
            df["Original Prices"]
            - df["Price"]
        )

    # Recalculate effective discount rate
    if (
        "Original Prices" in df.columns
        and "Price" in df.columns
    ):

        df["Discount_Rate_Calculated"] = np.where(
            df["Original Prices"] > 0,

            (
                (
                    df["Original Prices"]
                    - df["Price"]
                )
                /
                df["Original Prices"]
            ) * 100,

            0
        )

    # Round numeric columns
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

            df[column] = df[column].round(2)

    return df



# 13. CREATE SUMMARY


def create_summary(df):

    summary = (
        df.groupby("Category")
        .agg(
            Product_Count=(
                "Product Name",
                "count"
            ),

            Brand_Count=(
                "Brand",
                "nunique"
            ),

            Total_Price=(
                "Price",
                "sum"
            ),

            Total_Original_Price=(
                "Original Prices",
                "sum"
            ),

            Total_Discount=(
                "Discount_Amount",
                "sum"
            ),

            Average_Discount=(
                "Discount rates",
                "mean"
            )
        )
        .reset_index()
    )

    return summary.round(2)



# 14. SAVE DATA


def save_data(df, summary):

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    # Main cleaned dataset
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # Category summary
    summary_file = os.path.join(
        OUTPUT_FOLDER,
        "category_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print("\nFiles saved:")
    print(
        OUTPUT_FILE
    )
    print(
        summary_file
    )



# 15. MAIN PROGRAM


def main():

    print("=" * 60)
    print("FLIPKART DATA CLEANING")
    print("=" * 60)

    # Step 1
    # extract_dataset(
    #     ZIP_FILE,
    #     EXTRACT_FOLDER
    # )

    # Step 2
    csv_files = find_csv_files( "data/raw")

    print(
        f"\nCSV files found: {len(csv_files)}"
    )

    for file in csv_files:
        print(
            " -",
            os.path.basename(file)
        )

    # Step 3
    datasets = load_all_datasets(
        csv_files
    )

    # Step 4
    datasets = standardize_columns(
        datasets
    )

    # Step 5
    raw = combine_datasets(
        datasets
    )

    print(
        f"\nCombined rows: {len(raw)}"
    )

    # Step 6
    raw = final_cleaning(
        raw
    )

    # Step 7
    summary = create_summary(
        raw
    )

    # Step 8
    save_data(
        raw,
        summary
    )

  
    # Final information
    

    print("\n" + "=" * 60)
    print("FINAL DATASET")
    print("=" * 60)

    print(
        "Rows:",
        raw.shape[0]
    )

    print(
        "Columns:",
        raw.shape[1]
    )

    print("\nColumns:")
    print(
        raw.columns.tolist()
    )

    print("\nCategory Summary:")
    print(summary)

    print("\nMissing Values:")
    print(
        raw.isna().sum()
    )

    print("\nCleaning completed successfully.")



# 16. RUN PROGRAM


if __name__ == "__main__":
    main()