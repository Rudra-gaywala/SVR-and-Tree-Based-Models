# House Price Prediction -- Regression Models

## Project Overview

This project uses a house-price dataset containing 3,800 records to
predict `house_price_inr` (house price in Indian rupees). The notebook
explores regularized linear regression, cross-validation, tree-based
regression, and Support Vector Regression (SVR).

The work is implemented in the Jupyter Notebook `pr_2.ipynb`.

## Dataset

-   **Dataset:** Advanced Regression House Price Dataset
-   **Target column:** `house_price_inr`
-   **Rows:** 3,800 (as indicated by the dataset name)
-   **Input file used in the notebook:**
    `Advanced_Regression_HousePrice_Dataset_3800 - Advanced_Regression_HousePrice_Dataset_3800.csv.csv`

Make sure the CSV filename in the notebook matches the file available in
your project folder.

### Data preparation

The notebook: 1. Converts `sale_date` to a pandas datetime value. 2.
Extracts `sale_year` and `sale_month` from `sale_date`. 3. Removes
`sale_date`, `property_id`, `near_school`, `near_metro`, and
`crime_rate_index` from the predictor set. 4. Uses `house_price_inr` as
the target. 5. Splits the data into training and testing sets using an
80:20 split and `random_state=42`. 6. Applies `StandardScaler` to the
training and test features. It also scales the target values using the
scaler in the notebook.

The final feature matrix contains the remaining columns, including the
extracted `sale_year` and `sale_month`.

## Models Implemented

### 1. Ridge Regression

-   Uses `Ridge`.
-   Tunes `alpha` with `GridSearchCV`.
-   Tests alpha values: `0.01`, `0.1`, `1`, `10`, `100`, and `1000`.
-   Evaluates predictions using R² and mean squared error (MSE).

### 2. Lasso Regression

-   Uses `Lasso` with `max_iter=100000`.
-   Tunes `alpha` using the same parameter values and 5-fold grid
    search.
-   Evaluates predictions using MSE and R².

### 3. Cross-Validation Techniques

The notebook demonstrates:

-   **K-Fold:** 5 splits, shuffled, with `random_state=42`.
-   **Stratified K-Fold:** Creates five bins from the continuous target
    using `pd.qcut`, then uses those bins to create stratified folds.
-   **Leave-One-Out (LOOCV):** Uses one observation for each validation
    fold.
-   **Time Series Split:** Uses 5 sequential splits.

The notebook calculates cross-validation scores with negative mean
squared error for these examples.

### 4. Decision Tree Regression

-   Builds a default `DecisionTreeRegressor`.
-   Builds a second tree with `max_depth=5`, `min_samples_split=10`, and
    `min_samples_leaf=5`.
-   Compares the models using MAE, MSE, and R².

### 5. Random Forest Regression

-   Builds a `RandomForestRegressor` with 100 trees.
-   Builds a second random forest with `max_depth=5` and
    `min_samples_split=10`.
-   Compares the models using MAE, MSE, and R².

### 6. Support Vector Regression (SVR)

The notebook trains three SVR models:

  Model            Kernel     Additional setting
  ---------------- ---------- ----------------------
  Linear SVR       `linear`   Default SVR settings
  Polynomial SVR   `poly`     `degree=3`
  RBF SVR          `rbf`      Default SVR settings

The notebook compares these models using MAE, MSE, and R².

## Evaluation Metrics

-   **MAE (Mean Absolute Error):** Average absolute difference between
    actual and predicted values. Lower is better.
-   **MSE (Mean Squared Error):** Average squared prediction error.
    Lower is better.
-   **RMSE (Root Mean Squared Error):** Square root of MSE. Lower is
    better.
-   **R² score:** Measures the proportion of target variance explained
    by the model. Higher is generally better; it is not classification
    accuracy.

Because the notebook scales `y_train` and `y_test`, the reported
regression errors and predictions are in scaled target units, not
directly in INR.

## Requirements

Install the main Python packages:

``` bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## How to Run

1.  Place `pr_2.ipynb` and the dataset CSV in the same project folder.
2.  Check the `pd.read_csv(...)` filename in the notebook and update it
    if your CSV has a different name.
3.  Install the required packages.
4.  Open Jupyter Notebook or JupyterLab.
5.  Open `pr_2.ipynb`.
6.  Run the notebook cells from top to bottom.

## Project Structure

``` text
project-folder/
├── pr_2.ipynb
├── Advanced_Regression_HousePrice_Dataset_3800 - Advanced_Regression_HousePrice_Dataset_3800.csv.csv
└── README.md
```

## Notes and Limitations

-   The notebook demonstrates several regression approaches; it does not
    include a final saved model or a deployment interface.
-   The notebook's cross-validation examples use the original `x` and
    `y` variables, while the train/test models use scaled training and
    test data. For a rigorous comparison, preprocessing should be
    performed inside a scikit-learn `Pipeline` during cross-validation
    to avoid data leakage.
-   Stratified K-Fold is designed for classification. Binning a
    continuous target can be used to create stratified folds for
    regression, but the notebook currently passes the bin labels as the
    target in that particular Ridge cross-validation example. This
    evaluates prediction of bins rather than house prices.
-   Time Series Split is meaningful only when observations are ordered
    chronologically. Confirm the row order before interpreting those
    scores as forecasting performance.
-   For SVR, feature scaling is important. The notebook scales features
    before training the SVR models.
-   The notebook includes SVR model comparison, but the
    hyperparameter-tuning section should be run separately if you want
    tuned SVR results.

## Author

Add your name and contact details here.
