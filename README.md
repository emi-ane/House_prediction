🏠 House Price Prediction and Classification Project

📌 Project Overview

This project aims to accurately predict house prices and classify them into pricing categories (low, medium, high) using state-of-the-art regression and classification algorithms.
It leverages a cleaned and feature-engineered housing dataset to develop high-performing machine learning models through rigorous tuning, validation, and visualization.


🎯 Objectives

 -🔢 Predict house prices using regression models (Ridge, SVR, XGBoost Regressor)

 -🍿 Classify houses into pricing categories using classification models (Random Forest, LightGBM, XGBoost Classifier)

 -🔍 Handle data imbalance and feature skewness

 -⚙️ Perform hyperparameter tuning to improve model performance

 -📊 Visualize predictions and model decision boundaries


🛠️ Tools & Libraries Used

 -Python, NumPy, Pandas, Matplotlib, Seaborn

 -Scikit-learn, XGBoost, LightGBM, imblearn

 -t-SNE for dimensionality reduction


🧹 Data Preprocessing

 -✅ Removed irrelevant features

 -🧮 Applied log transformation to price column for regression

 -🛡️ Created new features (e.g., total bathrooms, property age)

 -🚫 Handled missing values

 -📊 Normalized numerical features for regressors

 -♻️ Balanced target classes using SMOTE for classification


 🧪 Models and Results

  - ⟳ Regression Models (Log-transformed price target)

| Model           | Train R²   | Test R²    | Test RMSE   | Notes                                   |
| --------------- | ---------- | ---------- | ----------- | --------------------------------------- |
| Ridge (Tuned)   | 0.9978     | 0.8423     | 530,839     | Excellent generalization                |
| SVR (Tuned)     | 0.8296     | 0.8409     | 210,000     | Slightly less fit than Ridge            |
| XGBoost (Tuned) | **0.9186** | **0.8397** | **210,000** | Best tradeoff of fit and generalization |

  ✅ XGBoost Regressor emerged as the most balanced and performant model for price prediction.


  -  🧠 Classification Models (Price categories: 0, 1, 2)
  
| Model                      | Accuracy   | Class 0 F1 | Class 1 F1 | Class 2 F1 | Notes                                 |
| -------------------------- | ---------- | ---------- | ---------- | ---------- | ------------------------------------- |
| Random Forest (Tuned)      | 0.8235     | 0.87       | 0.82       | 0.68       | Good balance across classes           |
| LightGBM (Tuned)           | 0.8200     | 0.86       | 0.82       | 0.67       | Slightly weaker on minority class     |
| XGBoost Classifier (Tuned) | **0.8321** | **0.87**   | **0.83**   | **0.74**   | **Best overall classification model** |

   ✅ XGBoost Classifier was the strongest performer in classification, particularly after SMOTE and hyperparameter tuning.


📉 Visualizations

  -🧫 t-SNE and PCA showed good class separation potential

  -📌 Confusion matrices revealed class 2 (high-price) was hardest to predict

  -🔍 Sample predictions demonstrated models' close approximation to real values


⚠️ Challenges & Fixes 💡

 -🧍‍♂️ Imbalanced Classes (Classification)

   -👉 Problem: Class 2 (high-price) had fewer examples
   -✅ Fix: Applied SMOTE to oversample the minority class

 -🧠 Overfitting in Untuned Models

   -👉 Problem: High training accuracy but poor generalization
   -✅ Fix: Tuned models using GridSearchCV to improve validation performance

 -⟳ Inconsistent Results When Rerunning Cells

   -👉 Problem: Variables retained from earlier runs interfered with new predictions
   -✅ Fix: Used %reset_selective and reran training cells in order


✅ Model Testing

Predicted samples from XGBoost Classifier and Regressor were compared with actuals and showed good alignment, demonstrating real-world predictive power.

-📋 Sample Results

  -Regression:
   Sample 1:
    Actual Price:    $535,000.00
    Predicted Price: $482,803.03
   Sample 2:
    Actual Price:    $445,000.00
    Predicted Price: $430,307.06

  -Classification:
  Sample 1:
   Actual Class:    1
   Predicted Class: 0
  Sample 2:
   Actual Class:    0
   Predicted Class: 0

💾 Conclusion

This project showcased a comprehensive approach to house price analysis through both regression and classification.

The best regression model was Tuned XGBoost Regressor, with a Test R² of 0.84 and lowest RMSE.

For classification, Tuned XGBoost Classifier slightly outperformed others with an accuracy of 83.2%.

Between regression and classification, regression is more informative when exact pricing is needed, while classification is useful for grouping into price ranges.

Thanks to extensive preprocessing, feature engineering, and model tuning, the final models achieved robust and reliable performance that can be confidently deployed
or further integrated into a pricing tool or real estate platform.


👩‍💻 Author
emi-ane



