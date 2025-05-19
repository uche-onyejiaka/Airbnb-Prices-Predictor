<<<<<<< HEAD
# Airbnb Prices Predictor (NYC Dataset)

This project uses machine learning models to predict Airbnb listing prices in New York City based on location, room type, availability, and other features. It helps Airbnb hosts make smarter pricing decisions by analyzing what truly impacts nightly rates.

---

## 📊 Project Overview

**Goal:**  
Predict nightly Airbnb prices using supervised learning models.

**Models Used:**  
- Linear Regression (baseline)
- Random Forest Regressor (final model)

**Key Features Selected:**  
- Latitude & Longitude
- Room Type (Entire home, Private room, Shared room)
- Minimum Nights
- Number of Reviews
- Reviews per Month
- Host Listing Count
- Availability (365 days)

---

## 🗂 Files Included

- `Airbnb_Price_Prediction.ipynb`: Jupyter Notebook containing all data cleaning, modeling, and analysis code.
- `listings.csv`: The original dataset from Inside Airbnb (if too large, download it [here](https://insideairbnb.com/get-the-data/)).
- `Top 20 Feature Importances.png`: Feature importance chart showing most influential factors.
- `README.md`: This file.

---

## 📚 Data Source

- Dataset: [Inside Airbnb - New York City Listings](https://insideairbnb.com/get-the-data/)
- Last updated: March 2025

---

## 🛠 Libraries and Tools

- **pandas**: For data manipulation
- **numpy**: For numerical operations
- **scikit-learn**: For machine learning models
- **matplotlib** & **seaborn**: For data visualization

---

## 📈 Key Results

| Model                  | RMSE ($) | MAE ($) | R² Score |
|------------------------|---------:|--------:|---------:|
| Linear Regression      | 100.06   | 50.11   | 0.07     |
| Random Forest Regressor|  95.10   | 42.16   | 0.16     |

✅ Random Forest outperformed linear regression by capturing more complex, non-linear relationships in the data.

✅ Top 5 features influencing price included location coordinates and review activity.

---

## ⚡ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/uche-onyejiaka/Airbnb-Prices-Predictor.git

Open Airbnb_Price_Prediction.ipynb in Jupyter Notebook.
Run all cells to reproduce the results.
🧠 Limitations

Missing amenity details (like hot tubs, rooftops) were not captured.
Seasonal pricing effects (e.g., Marathon Weekend surges) were not modeled.
Dataset is slightly biased toward listings in Manhattan and Brooklyn.
📬 Contact

If you have any questions, feel free to reach out via GitHub or LinkedIn!

Project created by Uche Onyejiaka as part of INST414 Supervised Learning assignment.


=======
# 🏡 SmartPrice: Airbnb Price Recommender

**SmartPrice** is a Streamlit-powered web app that helps Airbnb hosts determine the optimal nightly price for their listings based on real NYC Airbnb data and machine learning. Built using Python and Random Forest Regression, this tool offers personalized pricing recommendations and interactive data visualizations to empower hosts with data-driven decisions.

---

## 🎯 Project Motivation

Airbnb hosts — especially those new to the platform — often struggle to price their listings competitively. Pricing too high leads to empty bookings; pricing too low means leaving money on the table. The **core question** driving this project is:

> “How can Airbnb hosts confidently and competitively price their listings using real-world data?”

This tool is designed to answer that, giving hosts instant insights based on location, reviews, availability, and room type.

---

## 👤 Who This Is For

- **Airbnb Hosts**: Get a smart, personalized nightly price for your property.
- **Data Science Students**: Learn how to build a complete ML workflow with deployment.
- **Instructors/Graders**: This project demonstrates applied machine learning, feature engineering, model evaluation, and front-end integration.

---

## 📁 Features

✅ Predict nightly prices for NYC Airbnb listings  
✅ Explore data trends with boxplots, heatmaps, and bar charts  
✅ Clean, intuitive Streamlit user interface  
✅ Built with Random Forest Regression (sklearn)  
✅ Fully reproducible with open-source data  

---

## 🛠️ Technologies Used

- **Python 3.9**  
- **Streamlit**  
- **Pandas & NumPy**  
- **Matplotlib & Seaborn**  
- **Scikit-learn (Random Forest)**  
- Dataset: [`listings.csv`](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)

---

## 🚀 How to Run the App

### 1. Clone the Repository

```bash
git clone https://github.com/uche-onyejiaka/Airbnb-Prices-Predictor.git
cd Airbnb-Prices-Predictor

2. Install Dependencies

We recommend using a virtual environment (e.g. venv or conda), then run:
pip install -r requirements.txt

3. Add the Dataset

Make sure listings.csv is placed in the root folder of the project. You can get the file from Kaggle NYC Airbnb Dataset, or use the one already included.

4. Run the App

python3 -m streamlit run AirBnB_Price_Prediction.py

-You’ll be directed to http://localhost:8501 where you can interact with the app.

🧠 How It Works

The app loads and cleans real Airbnb listing data from New York City.
Key features like room type, neighborhood, review count, availability, and more are processed.
A Random Forest Regression model is trained to predict nightly price (with log transformation for skew correction).
The user inputs listing details into the app and receives a real-time price estimate.
Bonus: You can explore trends like top neighborhoods, pricing by room type, and review correlations in the "Explore Data" section.
📊 Example Use Case

A new host in Williamsburg wants to rent a private room for 3 nights with 20 reviews and high availability. By selecting their listing details in the app, they get an intelligent price recommendation tailored to their market.

📌 File Structure
├── AirBnB_Price_Prediction.py     # Main Streamlit application
├── listings.csv                   # NYC Airbnb dataset
├── requirements.txt               # Required Python packages
├── README.md                      # This file


🔒 Notes on Ethics & AI Use

This project was independently built using ethical data science practices. While ChatGPT was used for brainstorming and debugging, all final code, logic, modeling decisions, and design were my own. This app is meant to help Airbnb hosts, not exploit or misprice markets.

📎 License

This project is for educational and academic purposes. Attribution required for public or commercial use.
💬 Contact

Created by Uche Onyejiaka

https://github.com/uche-onyejiaka/Airbnb-Prices-Predictor
>>>>>>> 01ea00a (Final project files: Streamlit app, data, README, requirements)
