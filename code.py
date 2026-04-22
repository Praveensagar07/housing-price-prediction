import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ==========================================
# 1. LOAD AND PREPROCESS DATA
# ==========================================
def run_housing_analysis():
    print("--- Starting House Price Prediction Analysis ---")
    
    try:
        df = pd.read_csv('Housing.csv')
    except FileNotFoundError:
        print("Error: 'Housing.csv' not found. Please ensure the file is in the same directory.")
        return

    # Convert Binary categorical variables (yes/no) to (1/0)
    binary_vars = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for var in binary_vars:
        df[var] = df[var].map({'yes': 1, 'no': 0})

    # One-hot encode furnishingstatus (creates dummy variables)
    df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

    # ==========================================
    # 2. FEATURE SELECTION & DATA SPLITTING
    # ==========================================
    X = df.drop('price', axis=1)
    y = df['price']

    # Split: 80% Training, 20% Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling: Essential for Linear and Ridge Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==========================================
    # 3. ADVANCED MODEL TRAINING
    # ==========================================
    models = {
        "Multiple Linear Regression": LinearRegression(),
        "Ridge Regression (L2)": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    model_results = []
    trained_objects = {}

    print("\nTraining Models...")
    for name, model in models.items():
        # Scale data for regression models, use raw for tree models (though scaled works for both)
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        
        # Metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        model_results.append({
            "Model": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2 Score": round(r2, 4)
        })
        trained_objects[name] = model

    # Display Results Table
    results_df = pd.DataFrame(model_results).sort_values(by="R2 Score", ascending=False)
    print("\n--- Model Performance Summary ---")
    print(results_df.to_string(index=False))

    # ==========================================
    # 4. VISUALIZATION
    # ==========================================
    # Set style
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Feature Importance (using Random Forest)
    best_rf = trained_objects["Random Forest Regressor"]
    importances = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=True)
    importances.plot(kind='barh', ax=axes[0], color='skyblue')
    axes[0].set_title("Key Factors Driving House Prices", fontsize=14)
    axes[0].set_xlabel("Importance Score")

    # Plot 2: Actual vs Predicted (using Gradient Boosting)
    best_model = trained_objects["Gradient Boosting Regressor"]
    y_pred = best_model.predict(X_test_scaled)
    axes[1].scatter(y_test, y_pred, alpha=0.6, color='darkgreen')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    axes[1].set_title("Actual vs. Predicted Prices (Gradient Boosting)", fontsize=14)
    axes[1].set_xlabel("Actual Price")
    axes[1].set_ylabel("Predicted Price")

    plt.tight_layout()
    plt.show()

    # ==========================================
    # 5. PREDICTION EXAMPLE
    # ==========================================
    print("\n--- Manual Prediction Example ---")
    # Taking the first house in the test set as an example
    sample_house = X_test_scaled[0].reshape(1, -1)
    actual_val = y_test.iloc[0]
    predicted_val = trained_objects["Gradient Boosting Regressor"].predict(sample_house)[0]
    
    print(f"Actual Price: {actual_val:,.2f}")
    print(f"Predicted Price: {predicted_val:,.2f}")
    print(f"Difference: {abs(actual_val - predicted_val):,.2f}")

if __name__ == "__main__":
    run_housing_analysis()