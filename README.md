# Flipkart Analytics Project

##  Project Overview

**Flipkart Analytics** is an end-to-end **E-commerce Data Analytics and
Machine Learning project** built with Python. The project analyzes
Flipkart product data across multiple categories to identify patterns in
**product prices, original prices, discounts, ratings, brands, price
segments, and category-level performance**.

The project follows a complete data-science workflow:

**Raw Data → Data Cleaning → Exploratory Data Analysis → Feature
Engineering → Machine Learning → Model Evaluation → Streamlit
Dashboard**

The main goal is to transform raw and inconsistent e-commerce CSV data
into a clean, analysis-ready dataset and then provide useful business
insights through visualizations, machine-learning models, and an
interactive dashboard.

------------------------------------------------------------------------

##  Project Objectives

-   Clean and standardize multiple Flipkart product datasets.
-   Handle missing values and duplicate records.
-   Convert price and discount fields into usable numerical values.
-   Analyze products across different categories.
-   Compare selling price and original price.
-   Analyze discount rates and discount amounts.
-   Analyze product ratings where available.
-   Perform brand-level and category-level analysis.
-   Create engineered features for machine learning.
-   Prevent target leakage during model training.
-   Train and evaluate regression models.
-   Provide an interactive Streamlit dashboard.
-   Support dynamic product filtering based on price, rating, and
    discount.

------------------------------------------------------------------------

##  Dataset

### Dataset Contents

The project currently contains the following category-level CSV files:

``` text
baby.csv
books.csv
food.csv
furn.csv
laptops.csv
mens_westernwear.csv
mobiles.csv
women_footwear.csv
women_westernwear.csv
```

The combined dataset contains approximately **6,584 product records**.

### Dataset Link / Location

The dataset used by this project is included directly in the project
repository:

``` text
data/raw/
```

After cleaning, the combined dataset is available at:

``` text
processed_data/flipkart_cleaned.csv
```

### Important Dataset Fields

  Column              Description
  ------------------- -----------------------------------------------
  `Product Name`      Name of the product
  `Price`             Current/selling price
  `Original Prices`   Original/reference price
  `Discount rates`    Listed discount percentage
  `Brand`             Product brand, where available
  `Ratings`           Product rating, where available
  `Category`          Product category
  `Discount_Amount`   Difference between original and selling price

> **Note:** The dataset is a project dataset containing selected product
> records. It should not be interpreted as a complete real-time
> representation of the entire Flipkart marketplace.

------------------------------------------------------------------------

##  Project Structure

``` text
Flipkart-Analytics/
│
├── analysis_results/
│   ├── CSV analysis outputs
│   └── PNG charts
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   ├── baby.csv
│   │   ├── books.csv
│   │   ├── food.csv
│   │   ├── furn.csv
│   │   ├── laptops.csv
│   │   ├── mens_westernwear.csv
│   │   ├── mobiles.csv
│   │   ├── women_footwear.csv
│   │   └── women_westernwear.csv
│   │
│   └── processed/
│
├── feature_engineered_data/
│   ├── feature_summary.csv
│   ├── flipkart_feature_engineered.csv
│   └── flipkart_ml_ready.csv
│
├── model_results/
│   ├── saved_models/
│   ├── actual_vs_predicted.png
│   ├── model_comparison.csv
│   ├── model_predictions.csv
│   ├── residual_analysis.png
│   └── sample_predictions.csv
│
├── notebooks/
│
├── powerBI/
│
├── processed_data/
│   ├── category_summary.csv
│   └── flipkart_cleaned.csv
│
├── src/
│   ├── data_cleaning.py
│   ├── analysis.py
│   ├── feature_engineering.py
│   └── model.py
│
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

##  Technologies Used

### Programming Language

-   **Python 3**

### Data Processing

-   **Pandas**
-   **NumPy**

### Data Visualization

-   **Matplotlib**
-   **Seaborn**

### Machine Learning

-   **Scikit-learn**
-   **Joblib**

### Dashboard

-   **Streamlit**

### Development Environment

-   **Visual Studio Code**
-   Python Virtual Environment

### Data Format

-   CSV

------------------------------------------------------------------------

##  Project Workflow

``` text
                 ┌─────────────────┐
                 │   Raw CSV Data  │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Data Cleaning   │
                 │ data_cleaning.py│
                 └────────┬────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │ Cleaned Master Dataset │
              └────────────┬───────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
     ┌─────────────────┐       ┌────────────────────┐
     │ EDA / Analysis  │       │ Feature Engineering│
     │   analysis.py   │       │feature_engineering│
     └────────┬────────┘       └──────────┬─────────┘
              │                           │
              ▼                           ▼
     ┌─────────────────┐       ┌────────────────────┐
     │ Charts & CSV    │       │ ML Ready Dataset   │
     └─────────────────┘       └──────────┬─────────┘
                                          │
                                          ▼
                               ┌────────────────────┐
                               │ Machine Learning   │
                               │     model.py       │
                               └──────────┬─────────┘
                                          │
                                          ▼
                               ┌────────────────────┐
                               │ Model Results      │
                               └──────────┬─────────┘
                                          │
                                          ▼
                               ┌────────────────────┐
                               │ Streamlit Dashboard│
                               │      app.py        │
                               └────────────────────┘
```

------------------------------------------------------------------------

##  1. Data Cleaning

The `data_cleaning.py` script performs the initial preprocessing.

### Main operations

-   Load all raw CSV files.
-   Identify product categories.
-   Standardize column names.
-   Clean currency symbols and commas.
-   Convert prices into numerical values.
-   Convert discount percentages into numerical values.
-   Clean product ratings.
-   Normalize brand names.
-   Handle missing values.
-   Detect and remove duplicate rows.
-   Calculate discount amount.
-   Combine all category datasets.
-   Generate category-level summaries.

### Output

``` text
processed_data/flipkart_cleaned.csv
processed_data/category_summary.csv
```

------------------------------------------------------------------------

##  2. Exploratory Data Analysis

The `analysis.py` script performs detailed EDA.

### Analysis includes

-   Product count by category.
-   Average selling price.
-   Median selling price.
-   Minimum and maximum price.
-   Average discount.
-   Discount amount.
-   Price distributions.
-   Discount distributions.
-   Price distribution by category.
-   Discount distribution by category.
-   Original price vs selling price.
-   Price outlier detection using IQR.
-   Rating analysis.
-   Price vs rating relationship.
-   Correlation analysis.
-   Brand analysis.
-   Category and brand analysis.
-   Price-band analysis.
-   Dynamic product search.

### Example Business Questions

-   Which category has the highest average product price?
-   Which category has the highest average discount?
-   Which brands have the largest number of products?
-   How does selling price vary between categories?
-   What is the relationship between original price and selling price?
-   Which products satisfy a selected price, rating, and discount
    condition?

------------------------------------------------------------------------

##  3. Feature Engineering

The `feature_engineering.py` script prepares the dataset for machine
learning.

Typical engineered information includes:

-   Price-based features.
-   Original-price features.
-   Discount amount.
-   Discount percentage.
-   Rating-related features.
-   Category information.
-   Brand information.
-   Price bands.
-   Encoded categorical features.
-   Numerical ML features.

### Output

``` text
feature_engineered_data/
├── feature_summary.csv
├── flipkart_feature_engineered.csv
└── flipkart_ml_ready.csv
```

------------------------------------------------------------------------

##  4. Machine Learning

The `model.py` script performs the machine-learning stage.

### Pipeline

1.  Load ML-ready data.
2.  Identify the target variable.
3.  Separate features `X` and target `y`.
4.  Remove target leakage.
5.  Prepare numerical and categorical features.
6.  Split the data into training and testing sets.
7.  Train regression models.
8.  Generate predictions.
9.  Calculate evaluation metrics.
10. Compare models.
11. Save predictions and trained models.

### Target Leakage

Target leakage is explicitly considered in the project.

For example, if `Price` is the target:

``` python
X = df.drop(columns=["Price"])
y = df["Price"]
```

Features that directly contain or reconstruct the target should not be
used as predictors.

This prevents unrealistically strong model performance.

------------------------------------------------------------------------

##  Model Evaluation

The project can evaluate regression models using:

  Metric   Description
  -------- ------------------------------
  MAE      Mean Absolute Error
  MSE      Mean Squared Error
  RMSE     Root Mean Squared Error
  R²       Coefficient of Determination

The generated model comparison is stored in:

``` text
model_results/model_comparison.csv
```

Prediction results are stored in:

``` text
model_results/model_predictions.csv
```

Diagnostic charts include:

``` text
model_results/actual_vs_predicted.png
model_results/residual_analysis.png
```

------------------------------------------------------------------------

##  5. Streamlit Dashboard

The Streamlit application is located at:

``` text
app/app.py
```

The dashboard provides interactive analysis of the Flipkart dataset.

### Dashboard Features

#### Executive Overview

Displays key metrics such as:

-   Total products.
-   Number of categories.
-   Number of brands.
-   Average price.
-   Total selling-price value.
-   Average discount.

#### Category Analysis

Users can compare:

-   Product count.
-   Average price.
-   Median price.
-   Discount.
-   Brand count.
-   Ratings where available.

#### Brand Analysis

Provides:

-   Brand-level product counts.
-   Brand price analysis.
-   Brand discount analysis.
-   Category-to-brand analysis.

#### Price & Discount Analysis

Includes:

-   Price distributions.
-   Discount distributions.
-   Category comparisons.
-   Original price vs selling price.

#### Product Finder

Users can dynamically filter products using:

``` text
Category
Maximum Price
Minimum Rating
Minimum Discount
```

Example:

``` text
Category: Laptops
Maximum Price: ₹50,000
Minimum Rating: 4.0
Minimum Discount: 10%
```

------------------------------------------------------------------------

##  Installation

### 1. Clone or download the project

Place the project somewhere such as:

``` text
C:\Users\hp\Downloads\project_DA\Flipkart-Analytics-Project
```

Open the project folder in VS Code.

------------------------------------------------------------------------

### 2. Create a virtual environment

Windows:

``` bash
python -m venv .venv
```

------------------------------------------------------------------------

### 3. Activate the environment

Windows PowerShell:

``` bash
.venv\Scripts\Activate.ps1
```

Windows CMD:

``` bash
.venv\Scripts\activate
```

------------------------------------------------------------------------

### 4. Install dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

##  Run the Project

Run the modules in the following order.

### Step 1 --- Data Cleaning

``` bash
python src/data_cleaning.py
```

This generates:

``` text
processed_data/flipkart_cleaned.csv
processed_data/category_summary.csv
```

------------------------------------------------------------------------

### Step 2 --- Exploratory Data Analysis

``` bash
python src/analysis.py
```

This generates analysis tables and charts inside:

``` text
analysis_results/
```

------------------------------------------------------------------------

### Step 3 --- Feature Engineering

``` bash
python src/feature_engineering.py
```

This generates:

``` text
feature_engineered_data/
```

------------------------------------------------------------------------

### Step 4 --- Machine Learning

``` bash
python src/model.py
```

This generates:

``` text
model_results/
```

including model comparison, predictions, diagnostics and saved models.

------------------------------------------------------------------------

### Step 5 --- Start Streamlit

``` bash
streamlit run app/app.py
```

The Streamlit application will open in your browser.

------------------------------------------------------------------------

##  Requirements

The main Python packages are:

``` text
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
streamlit
```

Install all packages using:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

##  Important Output Files

  -----------------------------------------------------------------------------------------------
  File                                                        Purpose
  ----------------------------------------------------------- -----------------------------------
  `processed_data/flipkart_cleaned.csv`                       Cleaned master dataset

  `processed_data/category_summary.csv`                       Category-level summary

  `feature_engineered_data/flipkart_feature_engineered.csv`   Feature-engineered data

  `feature_engineered_data/flipkart_ml_ready.csv`             ML-ready data

  `feature_engineered_data/feature_summary.csv`               Feature information

  `analysis_results/`                                         EDA charts and CSV results

  `model_results/model_comparison.csv`                        Model performance

  `model_results/model_predictions.csv`                       Predictions

  `model_results/actual_vs_predicted.png`                     Actual vs predicted visualization

  `model_results/residual_analysis.png`                       Residual analysis

  `model_results/saved_models/`                               Serialized trained models
  -----------------------------------------------------------------------------------------------

------------------------------------------------------------------------

##  Key Insights

The dataset contains substantial differences between product categories.

Examples from the project dataset:

  Category                 Approx. Average Price   Approx. Average Discount
  ---------------------- ----------------------- --------------------------
  Baby                                      ₹621                      23.1%
  Books                                     ₹382                      23.1%
  Food                                      ₹394                      19.0%
  Furniture                               ₹5,173                      39.3%
  Laptops                                ₹62,042                      14.6%
  Men's Western Wear                        ₹447                      59.1%
  Mobiles                                 ₹9,958                      20.7%
  Women's Footwear                          ₹747                      39.2%
  Women's Western Wear                      ₹566                      59.7%

> These values describe the project's dataset and are not live Flipkart
> marketplace statistics.

------------------------------------------------------------------------

##  Business Use Cases

This project can be used to demonstrate:

-   E-commerce data analysis.
-   Product pricing analysis.
-   Discount analysis.
-   Brand performance analysis.
-   Category comparison.
-   Product discovery.
-   Data preprocessing.
-   Feature engineering.
-   Regression modeling.
-   Dashboard development.
-   Business intelligence workflows.

------------------------------------------------------------------------

##  Limitations

The current dataset does not provide all information required for
complete marketplace or seller profitability analysis.

For example, it does not reliably provide:

-   Cost of goods sold.
-   Seller commission.
-   Shipping cost.
-   Packaging cost.
-   Advertising cost.
-   Return cost.
-   Tax details.
-   Transaction-level sales volume.
-   Historical sales data.

Therefore, **actual seller profit or revenue cannot be calculated
reliably from product price alone**.

------------------------------------------------------------------------

##  Future Enhancements

Possible future improvements include:

-   Historical price tracking.
-   Sales-volume analysis.
-   Revenue forecasting.
-   Product recommendation system.
-   Seller-level analytics.
-   Inventory analysis.
-   Automated data refresh.
-   Advanced ML models.
-   Hyperparameter tuning.
-   SHAP/model explainability.
-   Automated model retraining.
-   Cloud deployment.
-   Power BI integration.
-   User authentication for the dashboard.

------------------------------------------------------------------------

##  Testing

Before running the ML pipeline, verify:

``` python
print(df.columns.tolist())
print(df.shape)
print(df.isnull().sum())
```

For a price-target model, verify:

``` python
assert "Price" in df.columns
```

After leakage removal, the target should still exist in the dataframe:

``` python
assert target in df.columns
```

The final feature matrix should not contain the target:

``` python
assert target not in X.columns
```

------------------------------------------------------------------------

##  Common Errors

### `KeyError: 'Price'`

This usually means that the dataframe loaded by the model does not
contain a column named `Price`.

Check:

``` python
print(df.columns.tolist())
```

Also verify that `model.py`, `feature_engineering.py`, and
`data_cleaning.py` are using the same dataset and column names.

------------------------------------------------------------------------

### `Target column 'Price' was accidentally removed`

This occurs when the leakage-removal function removes the target itself.

The correct concept is:

``` python
y = df["Price"]
```

and `Price` should be removed from the **feature matrix `X`**, not from
the source dataframe before `y` is created.

------------------------------------------------------------------------

### Streamlit Does Not Start

Try:

``` bash
python -m streamlit run app/app.py
```

instead of:

``` bash
streamlit run app/app.py
```

------------------------------------------------------------------------

##  Key Project Information

**Project Name:** Flipkart Analytics\
**Project Type:** E-commerce Data Analytics + Machine Learning\
**Language:** Python\
**Dashboard:** Streamlit\
**Data Format:** CSV\
**ML Type:** Regression\
**Visualization:** Matplotlib + Seaborn\
**Data Processing:** Pandas + NumPy\
**ML Framework:** Scikit-learn

------------------------------------------------------------------------

##  Author

**Sahil Mehta**

B.Tech Computer Science & Engineering

This project was developed as an end-to-end data analytics and
machine-learning portfolio project.

------------------------------------------------------------------------

##  License

This project is intended for **educational, academic, portfolio, and
demonstration purposes**.

If you redistribute the dataset or source data, verify the applicable
source/license terms separately.
