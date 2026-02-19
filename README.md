# What Drives the Price of a Used Car?

## Overview

This project analyzes a dataset of 426K used car listings (sourced from Kaggle) to identify the key factors that influence used car prices. The analysis follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework and delivers actionable recommendations for a used car dealership looking to optimize its inventory.

## Dataset

- **Source:** Kaggle (subset of 3 million listings)
- **Records:** 426,880 listings
- **Features (18 columns):**

| Feature | Type | Description |
|---|---|---|
| `id` | int | Unique listing identifier |
| `region` | string | Craigslist region of the listing |
| `price` | int | Listed sale price (target variable) |
| `year` | float | Model year of the vehicle |
| `manufacturer` | string | Vehicle manufacturer (42 unique) |
| `model` | string | Vehicle model (~29K unique) |
| `condition` | string | Vehicle condition (6 categories) |
| `cylinders` | string | Number of cylinders (8 categories) |
| `fuel` | string | Fuel type (5 categories) |
| `odometer` | float | Mileage on the vehicle |
| `title_status` | string | Title status (6 categories) |
| `transmission` | string | Transmission type (3 categories) |
| `VIN` | string | Vehicle Identification Number |
| `drive` | string | Drive type — 4wd, fwd, rwd |
| `size` | string | Vehicle size (4 categories) |
| `type` | string | Body type (13 categories) |
| `paint_color` | string | Exterior color (12 categories) |
| `state` | string | U.S. state (51 values) |

## Project Structure

```
practical_application_II_starter/
├── data/
│   └── vehicles.csv          # Raw dataset
├── images/
│   ├── kurt.jpeg
│   └── crisp.png
├── prompt_II.ipynb            # Main analysis notebook
└── README.md
```

## Methodology (CRISP-DM)

### 1. Business Understanding

The business objective is to identify the key drivers of used car prices so a dealership can make informed inventory decisions. This translates to a **supervised regression** problem where `price` is the target variable.

### 2. Data Understanding

- Explored data shape, types, and descriptive statistics
- Identified significant missing values in `size` (72%), `cylinders` (42%), `condition` (41%), `VIN` (38%), `drive` (31%), `paint_color` (30%), and `type` (22%)
- Visualized distributions of all numeric and categorical features
- Generated a correlation heatmap of numeric variables

### 3. Data Preparation

- **Dropped columns:** `id`, `VIN`, `region` (not predictive), `size` (too many nulls), `model` (too high cardinality)
- **Outlier removal:** Filtered prices to $500–$100,000; kept year >= 1990; capped odometer at 300,000 miles
- **Missing values:** Dropped rows missing critical fields (`year`, `manufacturer`, `odometer`, `fuel`, `transmission`, `title_status`); filled remaining categorical nulls with `"unknown"`
- **Feature engineering:** Created `vehicle_age` (2025 − year); converted `cylinders` strings to integers
- **Encoding:** One-hot encoded all categorical features (`manufacturer`, `condition`, `fuel`, `title_status`, `transmission`, `drive`, `type`, `paint_color`, `state`)
- **Scaling:** Applied `StandardScaler` to numeric features (`odometer`, `cylinders`, `vehicle_age`)
- **Train/test split:** 80/20 split with `random_state=42`

### 4. Modeling

| Model | Details | Test MSE |
|---|---|---|
| **Ridge Regression** | Default alpha | Baseline |
| **Ridge + GridSearchCV** | Alpha grid: [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0] | ~61.2M |

- Built a `Pipeline` with `StandardScaler` and `Ridge` regression
- Used `GridSearchCV` for hyperparameter tuning (best alpha = 100.0)
- Evaluated with **permutation importance** to rank feature contributions

### 5. Evaluation

- Compared train vs. test MSE to check for overfitting
- Analyzed permutation importance scores to identify the most influential features
- Key price drivers identified through feature importance analysis:
  - **Vehicle age** — strongest predictor (newer cars command higher prices)
  - **Odometer** — second strongest (lower mileage = higher price)
  - **Vehicle type** (pickup, truck, SUV vs. sedan)
  - **Manufacturer** (brand premium effects)
  - **Fuel type**, **drive type**, and **cylinders** also contribute

### 6. Deployment

Findings are presented in the Jupyter notebook as a report for the dealership client.

## Key Findings

1. **Vehicle age** is the single most important factor — newer vehicles are worth significantly more.
2. **Lower mileage** strongly correlates with higher prices.
3. **Body type matters** — pickups, trucks, and SUVs tend to hold value better than sedans.
4. **Brand premiums exist** — manufacturers like Toyota, Lexus, and Porsche command higher prices.
5. **Clean titles** are essential — salvage and rebuilt titles significantly reduce value.
6. **Fuel type and drive type** play secondary but meaningful roles.

## Technologies Used

- **Python 3.7**
- **pandas** — data manipulation
- **NumPy** — numerical operations
- **matplotlib / seaborn** — visualization
- **plotly** — interactive visualizations
- **scikit-learn** — modeling (Ridge regression, GridSearchCV, pipelines, permutation importance)
- **statsmodels** — statistical analysis

## How to Run

1. Ensure you have Python 3.7+ and the required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn plotly scikit-learn statsmodels
   ```
2. Open and run `prompt_II.ipynb` in Jupyter Notebook or VS Code.
3. The notebook reads data from `data/vehicles.csv` — ensure this file is present.
