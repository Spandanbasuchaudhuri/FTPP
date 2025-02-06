# **FTPP - Flight Ticket Price Prediction**

## **Project Overview**

Airline ticket prices can fluctuate significantly based on multiple factors such as:

- Date and time of travel
- Source and destination
- Number of stops
- Flight duration

This project aims to analyze, clean, and transform flight data and use machine learning techniques to predict ticket prices accurately.

### **Key Objectives**

✅ Perform exploratory data analysis (EDA) on flight fare data.
✅ Clean and preprocess the dataset by handling missing values and outliers.
✅ Engineer meaningful features like flight duration, total stops, etc.
✅ Build and evaluate regression models (**Linear Regression & Random Forest Regressor**) to predict ticket prices.

---

## **Dataset Overview**

- **File Name**: `Data_Train.xlsx`
- **Total Records**: **10,683 rows**, **11 columns**
- **Target Variable**: `Price` (flight fare)
- **Key Features**:
  - **Airline**: Name of the airline.
  - **Date\_of\_Journey**: Date on which the journey takes place.
  - **Source**: City from which the flight departs.
  - **Destination**: Arrival city.
  - **Route**: Path taken by the flight.
  - **Dep\_Time**: Flight departure time.
  - **Arrival\_Time**: Flight arrival time.
  - **Duration**: Duration of the flight.
  - **Total\_Stops**: Number of stops during the journey.
  - **Price**: Ticket price (target variable).

---

## **Installation & Dependencies**

### **Prerequisites**

- **Python 3.x**
- **NumPy, Pandas** (for data manipulation and analysis)
- **Matplotlib, Seaborn** (for data visualization)
- **scikit-learn** (for machine learning algorithms & evaluation)
- **Jupyter Notebook** or any Python IDE

### **Installation**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## **Data Preprocessing & Feature Engineering**

### **1. Handling Missing Values**

- `Route` and `Total_Stops` have missing values, handled by **dropping those rows**.

### **2. Feature Extraction & Transformation**

- **Date\_of\_Journey** converted to `datetime`, extracted into `Day` and `Month`.
- **Duration** converted from `2h 30m` format to total **minutes**.
- **Dep\_Time & Arrival\_Time** extracted into **hour** and **minute**.
- **Total\_Stops** mapped to numerical values.
- **Airline, Source, Destination** one-hot encoded.
- **Route** split into multiple steps and **label encoded**.

### **3. Scaling**

- Standardized numerical features using **StandardScaler**:
  ```python
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  data = scaler.fit_transform(data)
  ```

---

## **Machine Learning Models**

### **1. Linear Regression**

```python
from sklearn.linear_model import LinearRegression
model_lr = LinearRegression()
model_lr.fit(x_train, y_train)
```

### **2. Random Forest Regressor**

```python
from sklearn.ensemble import RandomForestRegressor
model_rf = RandomForestRegressor(n_estimators=500, min_samples_split=3)
model_rf.fit(x_train, y_train)
```

---

## **Model Evaluation**

| Model                 | RMSE       | R² Score   | Accuracy   |
| --------------------- | ---------- | ---------- | ---------- |
| **Linear Regression** | **0.5321** | **0.6514** | **72.54%** |
| **Random Forest**     | **0.3744** | **0.8274** | **97.42%** |

💡 **Insight:**\
The **Random Forest Regressor** significantly outperforms **Linear Regression**, achieving lower RMSE and higher accuracy.

---

## **Usage Instructions**

1. **Download the dataset** (`Data_Train.xlsx`).
2. **Install dependencies** (see the installation section).
3. **Run the Jupyter Notebook / Python script**:
   ```bash
   python main.py
   ```
4. **Interact with the Model**:
   - Load the trained model.
   - Input flight details.
   - Receive **predicted fare estimates**.

---

## **Conclusion**

### **Key Takeaways**

✅ **Feature engineering enhances model performance**, improving accuracy.\
✅ **Random Forest performs better** due to its ability to handle non-linear relationships.\
✅ **The model can be extended** using **Gradient Boosting, XGBoost, or Deep Learning**.

### **Future Enhancements**

🚀 Integrate **real-time web scraping** for live price predictions.\
🚀 Use **LSTMs (Neural Networks)** for time-series forecasting.\
🚀 Deploy the model using **Flask or FastAPI** for an interactive web app.

---

## **License**

This project is intended for educational and demonstration purposes. Feel free to modify and use the code as per your needs.

---

## **Contact**

For any questions, suggestions, or support, reach out:

- **Name**: Spandanbasuchaudhuri
- **Email**: [Spandanbasu139@gmail.com](mailto\:Spandanbasu139@gmail.com)
- **GitHub**: [Spandanbasuchaudhuri](https://github.com/Spandanbasuchaudhuri)

---

**Thank you for exploring this project!** 🚀 If you found this helpful, consider contributing or providing feedback.

