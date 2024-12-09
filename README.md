# Predicting Daily Bike Rental Availability

## Author
Ebhota Walter Eromosele

## Objective
Predict the number of rental bikes needed on the streets of Washington D.C. each day, based on weather conditions and other environmental factors.

### Importance of the Study
- **Resource Allocation**: Ensures the supply of bikes matches demand.
- **Customer Satisfaction**: Reduces shortages and improves user experience.
- **Strategic Planning**: Guides station placement and fleet expansion.

---

## Dataset Information
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/00275/)
- **Time Period**: January 2011 - December 2012
- **Attributes**:
  - Weather conditions (e.g., temperature, humidity, windspeed).
  - Temporal data (e.g., holiday, season).

### Key Insights from Exploration
- **Temperature Correlation**: Higher temperatures correspond to increased bike rentals.
- **No Missing Values**: The dataset is clean and ready for modeling.

---

## Models Used
1. **Random Forest Regressor**
   - Captures non-linear relationships and ranks feature importance.
   - Achieved **MAE: `<value will be printed>`**.

2. **Linear Regression**
   - Simple baseline model.
   - Achieved **MAE: `<value will be printed>`**.

---

## Project Setup and Execution

### Requirements
Install the required libraries:
```bash
pip install pandas requests seaborn matplotlib scikit-learn


---

### Steps to Run the Project in VS Code

1. **Set Up Project Folder**:
   - Create a folder named `BikeRentalPrediction` (or any name of your choice).
   - Save `bike_rental_prediction.py` and `README.md` in this folder.

2. **Open the Folder in VS Code**:
   - Launch VS Code and open the project folder.

3. **Install Dependencies**:
   - Open the terminal in VS Code and run:
     ```bash
     pip install pandas requests seaborn matplotlib scikit-learn
     ```

4. **Run the Python Script**:
   - In the terminal, run:
     ```bash
     python bike_rental_prediction.py
     ```

5. **View Results**:
   - The output will display in the terminal, and visualizations will pop up in a new window.




