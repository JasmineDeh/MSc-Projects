# Import necessary modules.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, confusion_matrix, classification_report, ConfusionMatrixDisplay

#################################################################################
# Washburn Equation from project brief.

def washburn_eq(gamma, r, t, phi, eta):
    """
    Compute penetration distance (L) using the Washburn equation.

    Args:
        gamma: Surface tension of liquid (N/m)
        r: Radius of pore (m)
        t: Time after initial dunking that the measurement was made (s)
        phi: Contact angle between the solid and liquid (rad)
        eta: Liquid viscosity (Pa s)

    Returns:
        L: Distance up the biscuit that the tea was visible (m)
    """
    return np.sqrt((gamma * r * t * np.cos(phi)) / (2 * eta))

#################################################################################
# Classifier and confusion matrix function. 

def classifier(X, y):
    """
    Applies classifier, returns confusion matrix.

    Args:
        X: Input independent variables.
        y: Output dependent variables.

    Returns:
        Accuracy score and confusion matrix.
    """
    # Train-test split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a classifier.
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)

    # Predictions.
    y_pred = classifier.predict(X_test)

    # Create confusion matrix.
    confus_mat = confusion_matrix(y_test, y_pred)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=confus_mat, display_labels=['Rich Tea', 'Hobnob', 'Digestive'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix for Biscuit Classification")
    plt.show()

    # Evaluate model.
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred, target_names=['Rich Tea', 'Hobnob', 'Digestive']))

#################################################################################
# Pore estimation function. 

def pore_rad_est(df):
    """
    Function to estimate pore radius using the Washburn equation.

    Args:
        df: Dataframe of interest.

    Returns:
        Amended Dataframe with pore radius estimation.
    """
    # Surface tension (N/m)
    gamma = 6.78e-2
    # Viscosity (Pa s)
    eta = 9.93e-4
    # Contact angle (rad)
    phi = 1.45
    # Applying Washburn eq 
    df['r_est'] = (2 * eta * (df['L']**2)) / (gamma * df['t'] * np.cos(phi))
    return df.head()


#################################################################################
# Function to compare regressor model and Washburn equation against actual data.

def regressor_washburn(X, y, df):
    """
    Applies random forest regressor, returns regressor and washburn errors

    Args:
        X: Input independent variables.
        y: Output dependent variables.
        df: Dataframe of interest.

    Returns:
        Regressor and washburn errors and plot to visually compare.
    """
    # Train-test split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a regressor.
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train, y_train)

    # Predictions
    y_pred = regressor.predict(X_test)

    # Washburn and regressor model mean squared errors
    washburn_error = mean_squared_error(y_test, df.loc[X_test.index, 'washburn_eq'])
    regressor_error = mean_squared_error(y_test, y_pred)

    print(f"Washburn Equation MSE: {washburn_error:.6f}")
    print(f"Regressor Model MSE: {regressor_error:.6f}")

    # Scatterplot to visualise.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['t'], df['L'], label="Actual Data", alpha=0.5)
    ax.scatter(df['t'], df['washburn_eq'], label="Washburn Prediction", alpha=0.5)
    ax.scatter(X_test['t'], y_pred, label="Regressor Prediction", alpha=0.5)
    ax.set(xlabel="Time (s)", ylabel="Penetration Distance (m)", title="Washburn Equation vs Regressor Model Predictions")
    ax.legend()
    plt.show()
