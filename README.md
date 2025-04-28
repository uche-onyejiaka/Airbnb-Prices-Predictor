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


