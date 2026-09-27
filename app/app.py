
# FLIPKART ANALYTICS DASHBOARD
# Streamlit Application


import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")



# PAGE CONFIGURATION


st.set_page_config(
    page_title="Flipkart Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)



# PROJECT PATHS


# app/app.py
#     ↓
# project root = parent of app folder

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Possible data locations
PROCESSED_DIR = PROJECT_ROOT / "processed_data"
FEATURE_DIR = PROJECT_ROOT / "feature_engineered_data"
MODEL_DIR = PROJECT_ROOT / "model_results"
SAVED_MODELS_DIR = MODEL_DIR / "saved_models"
ANALYSIS_DIR = PROJECT_ROOT / "analysis_results"



# CUSTOM CSS


st.markdown(
    """
    <style>

    .main-title {
        font-size: 40px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        background-color: #fafafa;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# HELPER FUNCTIONS


def find_file(filename):
    """
    Search the project for a specific file.
    """

    possible_locations = [
        PROCESSED_DIR / filename,
        FEATURE_DIR / filename,
        MODEL_DIR / filename,
        PROJECT_ROOT / filename
    ]

    for path in possible_locations:

        if path.exists():
            return path

    return None


def find_column(df, possible_names):
    """
    Find a column even if its spelling/capitalization differs.
    """

    column_map = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for name in possible_names:

        key = str(name).strip().lower()

        if key in column_map:
            return column_map[key]

    return None


def money_format(value):
    """
    Format numbers as Indian currency.
    """

    if pd.isna(value):
        return "₹0"

    return f"₹{value:,.0f}"


def percentage_format(value):
    """
    Format percentage.
    """

    if pd.isna(value):
        return "0%"

    return f"{value:.2f}%"



# LOAD DATA


@st.cache_data
def load_data():

    # First preference: cleaned data
    cleaned_file = PROCESSED_DIR / "flipkart_cleaned.csv"

    # Second preference: feature engineered data
    feature_file = FEATURE_DIR / "flipkart_feature_engineered.csv"

    if cleaned_file.exists():

        df = pd.read_csv(cleaned_file)

        source = "flipkart_cleaned.csv"

    elif feature_file.exists():

        df = pd.read_csv(feature_file)

        source = "flipkart_feature_engineered.csv"

    else:

        raise FileNotFoundError(
            """
            No dataset found.

            Expected one of:

            processed_data/flipkart_cleaned.csv

            or

            feature_engineered_data/flipkart_feature_engineered.csv
            """
        )

    return df, source



# LOAD DATA


try:

    df, data_source = load_data()

except Exception as e:

    st.error(
        f"Unable to load dataset:\n\n{e}"
    )

    st.stop()



# STANDARDIZE COLUMN NAMES


df.columns = [
    str(col).strip()
    for col in df.columns
]



# IDENTIFY IMPORTANT COLUMNS


CATEGORY_COL = find_column(
    df,
    [
        "Category",
        "category"
    ]
)

PRODUCT_COL = find_column(
    df,
    [
        "Product Name",
        "Product",
        "product"
    ]
)

BRAND_COL = find_column(
    df,
    [
        "Brand",
        "Brand_Normalized",
        "brand"
    ]
)

PRICE_COL = find_column(
    df,
    [
        "Price",
        "price",
        "Selling Price",
        "Selling_Price"
    ]
)

ORIGINAL_PRICE_COL = find_column(
    df,
    [
        "Original Prices",
        "Original Price",
        "Original_Price",
        "Market Price",
        "Market_Price"
    ]
)

DISCOUNT_COL = find_column(
    df,
    [
        "Discount rates",
        "Discount Rate",
        "Discount",
        "Discount_Percentage"
    ]
)

RATING_COL = find_column(
    df,
    [
        "Rating",
        "Ratings",
        "rating"
    ]
)



# CONVERT NUMERIC COLUMNS


for col in [
    PRICE_COL,
    ORIGINAL_PRICE_COL,
    DISCOUNT_COL,
    RATING_COL
]:

    if col is not None:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )



# SIDEBAR


st.sidebar.title("🛒 Flipkart Analytics")

st.sidebar.markdown(
    "### Dashboard Controls"
)

st.sidebar.info(
    f"Dataset: {data_source}"
)

st.sidebar.write(
    f"Rows: **{len(df):,}**"
)

st.sidebar.write(
    f"Columns: **{len(df.columns)}**"
)



# CATEGORY FILTER


filtered_df = df.copy()

if CATEGORY_COL is not None:

    categories = sorted(
        filtered_df[CATEGORY_COL]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_categories = st.sidebar.multiselect(
        "Select Category",
        categories,
        default=categories
    )

    if selected_categories:

        filtered_df = filtered_df[
            filtered_df[CATEGORY_COL]
            .astype(str)
            .isin(selected_categories)
        ]



# PRICE FILTER


if PRICE_COL is not None:

    min_price = float(
        filtered_df[PRICE_COL]
        .min()
        if not filtered_df[PRICE_COL].dropna().empty
        else 0
    )

    max_price = float(
        filtered_df[PRICE_COL]
        .max()
        if not filtered_df[PRICE_COL].dropna().empty
        else 100000
    )

    if min_price < max_price:

        price_range = st.sidebar.slider(
            "Price Range (₹)",
            min_value=float(min_price),
            max_value=float(max_price),
            value=(
                float(min_price),
                float(max_price)
            )
        )

        filtered_df = filtered_df[
            filtered_df[PRICE_COL]
            .between(
                price_range[0],
                price_range[1]
            )
        ]



# DISCOUNT FILTER


if DISCOUNT_COL is not None:

    discount_values = filtered_df[
        DISCOUNT_COL
    ].dropna()

    if not discount_values.empty:

        min_discount = float(
            discount_values.min()
        )

        max_discount = float(
            discount_values.max()
        )

        if min_discount < max_discount:

            discount_range = st.sidebar.slider(
                "Discount (%)",
                min_value=float(min_discount),
                max_value=float(max_discount),
                value=(
                    float(min_discount),
                    float(max_discount)
                )
            )

            filtered_df = filtered_df[
                filtered_df[DISCOUNT_COL]
                .between(
                    discount_range[0],
                    discount_range[1]
                )
            ]



# MAIN TITLE


st.markdown(
    '<div class="main-title">🛒 Flipkart Analytics Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Product, Category, Price, Discount, Brand and ML Analysis'
    '</div>',
    unsafe_allow_html=True
)


st.success(
    f"Showing {len(filtered_df):,} products"
)



# KPI SECTION


st.subheader("📊 Key Performance Indicators")


kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)


# Total Products

with kpi1:

    st.metric(
        "Total Products",
        f"{len(filtered_df):,}"
    )


# Total Categories

with kpi2:

    if CATEGORY_COL:

        total_categories = (
            filtered_df[CATEGORY_COL]
            .nunique()
        )

    else:

        total_categories = 0

    st.metric(
        "Categories",
        f"{total_categories:,}"
    )


# Average Price

with kpi3:

    if PRICE_COL:

        avg_price = filtered_df[
            PRICE_COL
        ].mean()

    else:

        avg_price = 0

    st.metric(
        "Average Price",
        money_format(avg_price)
    )


# Average Discount

with kpi4:

    if DISCOUNT_COL:

        avg_discount = filtered_df[
            DISCOUNT_COL
        ].mean()

    else:

        avg_discount = 0

    st.metric(
        "Avg Discount",
        percentage_format(avg_discount)
    )


# Average Rating

with kpi5:

    if RATING_COL:

        avg_rating = filtered_df[
            RATING_COL
        ].mean()

    else:

        avg_rating = 0

    st.metric(
        "Average Rating",
        f"{avg_rating:.2f}"
    )



# TABS


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📊 Overview",
        "📂 Category Analysis",
        "🏷️ Brand Analysis",
        "💰 Price & Discount",
        "🔎 Product Finder",
        "🤖 ML Results"
    ]
)



# TAB 1 — OVERVIEW


with tab1:

    st.header("Overall Dataset Analysis")

    col1, col2 = st.columns(2)

    #
    # Products by Category
    #

    with col1:

        st.subheader(
            "Products by Category"
        )

        if CATEGORY_COL:

            category_count = (
                filtered_df[
                    CATEGORY_COL
                ]
                .value_counts()
                .head(15)
            )

            fig, ax = plt.subplots(
                figsize=(9, 6)
            )

            category_count.sort_values().plot(
                kind="barh",
                ax=ax
            )

            ax.set_xlabel(
                "Number of Products"
            )

            ax.set_ylabel(
                "Category"
            )

            ax.set_title(
                "Top Categories by Product Count"
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)


    #
    # Price Distribution
    #

    with col2:

        st.subheader(
            "Price Distribution"
        )

        if PRICE_COL:

            fig, ax = plt.subplots(
                figsize=(9, 6)
            )

            sns.histplot(
                filtered_df[PRICE_COL]
                .dropna(),
                bins=30,
                kde=True,
                ax=ax
            )

            ax.set_xlabel(
                "Price (₹)"
            )

            ax.set_ylabel(
                "Number of Products"
            )

            ax.set_title(
                "Product Price Distribution"
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)


    #
    # Discount Distribution
    #

    if DISCOUNT_COL:

        st.subheader(
            "Discount Distribution"
        )

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        sns.histplot(
            filtered_df[DISCOUNT_COL]
            .dropna(),
            bins=30,
            kde=True,
            ax=ax
        )

        ax.set_xlabel(
            "Discount (%)"
        )

        ax.set_ylabel(
            "Products"
        )

        ax.set_title(
            "Distribution of Discounts"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)



# TAB 2 — CATEGORY ANALYSIS


with tab2:

    st.header(
        "📂 Category-Level Analysis"
    )

    if CATEGORY_COL:

        category_analysis = (
            filtered_df
            .groupby(CATEGORY_COL)
            .agg(
                Products=(
                    PRODUCT_COL,
                    "count"
                )
                if PRODUCT_COL
                else (
                    CATEGORY_COL,
                    "count"
                )
            )
        )

        if PRICE_COL:

            price_summary = (
                filtered_df
                .groupby(CATEGORY_COL)[PRICE_COL]
                .agg(
                    Total_Price="sum",
                    Average_Price="mean"
                )
            )

            category_analysis = (
                category_analysis
                .join(price_summary)
            )

        if ORIGINAL_PRICE_COL:

            original_summary = (
                filtered_df
                .groupby(CATEGORY_COL)[
                    ORIGINAL_PRICE_COL
                ]
                .sum()
                .rename(
                    "Total_Original_Price"
                )
            )

            category_analysis = (
                category_analysis
                .join(original_summary)
            )

        if DISCOUNT_COL:

            discount_summary = (
                filtered_df
                .groupby(CATEGORY_COL)[
                    DISCOUNT_COL
                ]
                .mean()
                .rename(
                    "Average_Discount"
                )
            )

            category_analysis = (
                category_analysis
                .join(discount_summary)
            )

        category_analysis = (
            category_analysis
            .sort_values(
                "Products",
                ascending=False
            )
        )

        st.subheader(
            "Category Summary"
        )

        st.dataframe(
            category_analysis,
            use_container_width=True
        )


        # Category Product Count

        st.subheader(
            "Products in Each Category"
        )

        fig, ax = plt.subplots(
            figsize=(12, 7)
        )

        category_analysis[
            "Products"
        ].sort_values().plot(
            kind="barh",
            ax=ax
        )

        ax.set_xlabel(
            "Number of Products"
        )

        ax.set_ylabel(
            "Category"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


        # Category Average Price

        if "Average_Price" in category_analysis:

            st.subheader(
                "Average Price by Category"
            )

            price_chart = (
                category_analysis[
                    "Average_Price"
                ]
                .sort_values(
                    ascending=False
                )
            )

            fig, ax = plt.subplots(
                figsize=(12, 7)
            )

            price_chart.sort_values().plot(
                kind="barh",
                ax=ax
            )

            ax.set_xlabel(
                "Average Price (₹)"
            )

            ax.set_ylabel(
                "Category"
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)



# TAB 3 — BRAND ANALYSIS


with tab3:

    st.header(
        "🏷️ Brand Analysis"
    )

    if BRAND_COL:

        brand_count = (
            filtered_df[BRAND_COL]
            .value_counts()
            .head(20)
        )

        st.subheader(
            "Top 20 Brands by Product Count"
        )

        fig, ax = plt.subplots(
            figsize=(12, 7)
        )

        brand_count.sort_values().plot(
            kind="barh",
            ax=ax
        )

        ax.set_xlabel(
            "Number of Products"
        )

        ax.set_ylabel(
            "Brand"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


        # Brand Price

        if PRICE_COL:

            brand_price = (
                filtered_df
                .groupby(BRAND_COL)[
                    PRICE_COL
                ]
                .agg(
                    Products="count",
                    Total_Price="sum",
                    Average_Price="mean"
                )
                .query(
                    "Products >= 5"
                )
                .sort_values(
                    "Total_Price",
                    ascending=False
                )
                .head(20)
            )

            st.subheader(
                "Top Brands by Total Product Price"
            )

            st.dataframe(
                brand_price,
                use_container_width=True
            )


        # Brand Discount

        if DISCOUNT_COL:

            brand_discount = (
                filtered_df
                .groupby(BRAND_COL)[
                    DISCOUNT_COL
                ]
                .agg(
                    Products="count",
                    Average_Discount="mean"
                )
                .query(
                    "Products >= 5"
                )
                .sort_values(
                    "Average_Discount",
                    ascending=False
                )
                .head(20)
            )

            st.subheader(
                "Brands with Highest Average Discount"
            )

            st.dataframe(
                brand_discount,
                use_container_width=True
            )



# TAB 4 — PRICE & DISCOUNT ANALYSIS


with tab4:

    st.header(
        "💰 Price & Discount Analysis"
    )

    col1, col2 = st.columns(2)


    
    # Price vs Discount
    

    with col1:

        if PRICE_COL and DISCOUNT_COL:

            st.subheader(
                "Price vs Discount"
            )

            plot_data = filtered_df[
                [
                    PRICE_COL,
                    DISCOUNT_COL
                ]
            ].dropna()

            fig, ax = plt.subplots(
                figsize=(9, 6)
            )

            sns.scatterplot(
                data=plot_data,
                x=DISCOUNT_COL,
                y=PRICE_COL,
                alpha=0.6,
                ax=ax
            )

            ax.set_xlabel(
                "Discount (%)"
            )

            ax.set_ylabel(
                "Selling Price (₹)"
            )

            ax.set_title(
                "Price vs Discount"
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)


    
    # Original Price vs Selling Price
    

    with col2:

        if PRICE_COL and ORIGINAL_PRICE_COL:

            st.subheader(
                "Original Price vs Selling Price"
            )

            plot_data = filtered_df[
                [
                    PRICE_COL,
                    ORIGINAL_PRICE_COL
                ]
            ].dropna()

            fig, ax = plt.subplots(
                figsize=(9, 6)
            )

            sns.scatterplot(
                data=plot_data,
                x=ORIGINAL_PRICE_COL,
                y=PRICE_COL,
                alpha=0.6,
                ax=ax
            )

            ax.set_xlabel(
                "Original Price (₹)"
            )

            ax.set_ylabel(
                "Selling Price (₹)"
            )

            ax.set_title(
                "Original Price vs Selling Price"
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)


    
    # Biggest Discount Products
    

    if DISCOUNT_COL:

        st.subheader(
            "🔥 Products with Highest Discount"
        )

        columns_to_show = []

        for col in [
            PRODUCT_COL,
            CATEGORY_COL,
            BRAND_COL,
            PRICE_COL,
            ORIGINAL_PRICE_COL,
            DISCOUNT_COL,
            RATING_COL
        ]:

            if col and col not in columns_to_show:

                columns_to_show.append(col)

        top_discount = (
            filtered_df[
                columns_to_show
            ]
            .sort_values(
                DISCOUNT_COL,
                ascending=False
            )
            .head(20)
        )

        st.dataframe(
            top_discount,
            use_container_width=True
        )



# TAB 5 — PRODUCT FINDER


with tab5:

    st.header(
        "🔎 Product Finder"
    )

    st.write(
        """
        Find products using dynamic price, rating and discount
        conditions.
        """
    )


    col1, col2, col3 = st.columns(3)


    # Dynamic maximum price

    with col1:

        if PRICE_COL:

            max_price_filter = st.number_input(
                "Maximum Price (₹)",
                min_value=0.0,
                value=50000.0,
                step=1000.0
            )

        else:

            max_price_filter = 50000


    # Dynamic minimum rating

    with col2:

        if RATING_COL:

            min_rating_filter = st.number_input(
                "Minimum Rating",
                min_value=0.0,
                max_value=5.0,
                value=4.0,
                step=0.1
            )

        else:

            min_rating_filter = 0


    # Dynamic minimum discount

    with col3:

        if DISCOUNT_COL:

            min_discount_filter = st.number_input(
                "Minimum Discount (%)",
                min_value=0.0,
                value=10.0,
                step=1.0
            )

        else:

            min_discount_filter = 0


    
    # Search Button
    

    if st.button(
        "🔍 Find Products",
        type="primary"
    ):

        result = filtered_df.copy()


        if PRICE_COL:

            result = result[
                result[PRICE_COL]
                <= max_price_filter
            ]


        if RATING_COL:

            result = result[
                result[RATING_COL]
                >= min_rating_filter
            ]


        if DISCOUNT_COL:

            result = result[
                result[DISCOUNT_COL]
                >= min_discount_filter
            ]


        st.success(
            f"{len(result):,} products found"
        )


        columns_to_show = []

        for col in [
            PRODUCT_COL,
            CATEGORY_COL,
            BRAND_COL,
            PRICE_COL,
            ORIGINAL_PRICE_COL,
            DISCOUNT_COL,
            RATING_COL
        ]:

            if col and col not in columns_to_show:

                columns_to_show.append(col)


        if len(result) > 0:

            st.dataframe(
                result[
                    columns_to_show
                ]
                .sort_values(
                    PRICE_COL
                    if PRICE_COL
                    else columns_to_show[0]
                ),
                use_container_width=True
            )

        else:

            st.warning(
                "No products match these conditions."
            )



# TAB 6 — MACHINE LEARNING RESULTS


with tab6:

    st.header(
        "🤖 Machine Learning Results"
    )

    comparison_file = (
        MODEL_DIR
        / "model_comparison.csv"
    )

    predictions_file = (
        MODEL_DIR
        / "model_predictions.csv"
    )


    
    # Model Comparison
    

    if comparison_file.exists():

        st.subheader(
            "Model Comparison"
        )

        comparison_df = pd.read_csv(
            comparison_file
        )

        st.dataframe(
            comparison_df,
            use_container_width=True
        )


        # R² chart

        r2_col = find_column(
            comparison_df,
            [
                "R2",
                "R²",
                "R2_Score",
                "R_squared"
            ]
        )

        model_col = find_column(
            comparison_df,
            [
                "Model",
                "model"
            ]
        )

        if r2_col and model_col:

            st.subheader(
                "Model R² Comparison"
            )

            chart_data = (
                comparison_df
                .set_index(model_col)[r2_col]
            )

            st.bar_chart(
                chart_data
            )


    else:

        st.info(
            "model_comparison.csv was not found."
        )


    
    # Predictions
    

    if predictions_file.exists():

        st.subheader(
            "Model Predictions"
        )

        predictions_df = pd.read_csv(
            predictions_file
        )

        st.dataframe(
            predictions_df.head(100),
            use_container_width=True
        )

    else:

        st.info(
            "model_predictions.csv was not found."
        )


    
    # Existing Model Images
    

    actual_predicted_image = (
        MODEL_DIR
        / "actual_vs_predicted.png"
    )

    residual_image = (
        MODEL_DIR
        / "residual_analysis.png"
    )


    col1, col2 = st.columns(2)


    with col1:

        if actual_predicted_image.exists():

            st.subheader(
                "Actual vs Predicted"
            )

            st.image(
                str(actual_predicted_image),
                use_container_width=True
            )


    with col2:

        if residual_image.exists():

            st.subheader(
                "Residual Analysis"
            )

            st.image(
                str(residual_image),
                use_container_width=True
            )



# DATA EXPLORER


st.divider()

st.header(
    "📋 Data Explorer"
)

show_data = st.checkbox(
    "Show filtered dataset"
)

if show_data:

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=500
    )



# DOWNLOAD DATA


st.subheader(
    "⬇️ Download Filtered Data"
)

csv_data = filtered_df.to_csv(
    index=False
)

st.download_button(
    label="Download CSV",
    data=csv_data,
    file_name="flipkart_filtered_data.csv",
    mime="text/csv"
)


# FOOTER


st.divider()

st.caption(
    "Flipkart Analytics Project | Python • Pandas • NumPy • "
    "Matplotlib • Seaborn • Scikit-learn • Streamlit"
)