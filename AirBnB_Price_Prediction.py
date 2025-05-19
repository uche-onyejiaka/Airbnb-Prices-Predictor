import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Configure the Streamlit page
st.set_page_config(page_title="SmartPrice: Airbnb Price Recommender", layout="wide")

# Title and intro
st.title("🏠 SmartPrice: Airbnb Price Recommender")
st.markdown("Welcome to SmartPrice — an intelligent tool that helps Airbnb hosts set the perfect price for their listing using real-world data and machine learning.")

# Load the Airbnb dataset
@st.cache_data
def load_data():
    return pd.read_csv("listings.csv")

df = load_data()

# Clean and preprocess the data
df = df.dropna(subset=["price"])
df = df[(df["price"] > 0) & (df["price"] < 1000)]  # Remove invalid or extreme prices
df = df.drop(columns=["neighbourhood_group", "license", "host_name", "last_review"])
df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
df["price_per_night"] = df["price"] / df["minimum_nights"]
df["log_price"] = np.log1p(df["price"])  # Log-transform helps with skewed price data

# Build the model
df_model = pd.get_dummies(df, columns=["room_type", "neighbourhood"], drop_first=True)
X = df_model.drop(columns=["id", "name", "host_id", "price", "log_price"])
y = df_model["log_price"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Sidebar for navigation between sections
section = st.sidebar.radio("Choose a section to explore:", ["🔍 Explore the Data", "💰 Predict Price"])

# SECTION 1: Data Exploration
if section == "🔍 Explore the Data":
    st.header("📊 Data Visualizations")
    st.markdown("Let’s take a look at what the Airbnb data tells us.")

    # 1. Room type vs price
    st.subheader("1. Room Type vs. Price Distribution")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="room_type", y="price", ax=ax1)
    ax1.set_title("Room Type vs. Price Distribution")
    st.pyplot(fig1)

    # 2. Average price per neighborhood
    st.subheader("2. Top 10 Neighbourhoods by Average Price")
    avg_price_neighbourhood = df.groupby("neighbourhood")["price"].mean().sort_values(ascending=False).head(10)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=avg_price_neighbourhood.values, y=avg_price_neighbourhood.index, ax=ax2)
    ax2.set_title("Top 10 Neighbourhoods by Average Price")
    st.pyplot(fig2)

    # 3. Reviews vs price
    st.subheader("3. Number of Reviews vs. Price")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="number_of_reviews", y="price", alpha=0.6, ax=ax3)
    ax3.set_title("Number of Reviews vs. Price")
    st.pyplot(fig3)

    # 4. Model feature importances
    st.subheader("4. What Factors Influence Price the Most?")
    importances = model.feature_importances_
    feature_names = X.columns
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(10)
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=ax4)
    ax4.set_title("Top 10 Feature Importances from the Model")
    st.pyplot(fig4)

# SECTION 2: Price Prediction
elif section == "💰 Predict Price":
    st.header("💸 Price Recommendation Engine")
    st.markdown("Fill out your listing details below to get a smart price suggestion.")

    # Collect user input
    room_type = st.selectbox("Room Type", df["room_type"].unique())
    neighbourhood = st.selectbox("Neighbourhood", df["neighbourhood"].unique())
    minimum_nights = st.number_input("Minimum Nights", min_value=1, value=2)
    number_of_reviews = st.slider("Number of Reviews", 0, 500, 10)
    reviews_per_month = st.slider("Reviews per Month", 0.0, 10.0, 1.0)
    availability_365 = st.slider("Availability (days per year)", 0, 365, 180)
    number_of_reviews_ltm = st.slider("Reviews in Last 12 Months", 0, 300, 5)
    calculated_host_listings_count = st.slider("Other Listings by Host", 1, 50, 2)

    # Build input feature dictionary
    input_dict = {
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "reviews_per_month": reviews_per_month,
        "calculated_host_listings_count": calculated_host_listings_count,
        "availability_365": availability_365,
        "number_of_reviews_ltm": number_of_reviews_ltm,
    }

    # One-hot encode user input to match training data format
    for col in X.columns:
        if "room_type_" in col:
            input_dict[col] = 1 if col.split("room_type_")[1] == room_type else 0
        elif "neighbourhood_" in col:
            input_dict[col] = 1 if col.split("neighbourhood_")[1] == neighbourhood else 0
        elif col not in input_dict:
            input_dict[col] = 0

    # Convert to DataFrame and predict
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=X.columns, fill_value=0)  # Ensure correct column order and fill missing
    log_price_pred = model.predict(input_df)[0]
    predicted_price = np.expm1(log_price_pred)

    # Display the result
    st.subheader(f"💰 Recommended Nightly Price: **${predicted_price:.2f}**")
    st.caption("This prediction is based on a machine learning model trained on actual NYC Airbnb listings.")
