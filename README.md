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

The business objective is to identify the key drivers of used car prices so a dealership can make informed inventory decisions. This translates to a **supervised regression** problem where `price` is the continuous target variable and vehicle attributes are the features.

### 2. Data Understanding

- Explored data shape, types, and descriptive statistics
- Identified significant missing values in `size` (72%), `cylinders` (42%), `condition` (41%), `VIN` (38%), `drive` (31%), `paint_color` (30%), and `type` (22%)
- Visualized distributions of all numeric and categorical features using Matplotlib
- Generated correlation heatmaps of numeric variables using Seaborn

### 3. Data Preparation

- **Dropped columns:** `id`, `VIN`, `region`, `state` (not predictive or too high cardinality), `size` (too many nulls), `model` (too high cardinality)
- **Outlier removal:** Used IQR-based detection and removal on `price`, `odometer`, and `cylinders`; filtered prices > 0; kept year >= 1990
- **Missing values:** Dropped all rows missing any feature column (no 'unknown' fill) to eliminate noise
- **Category filtering:** Removed rare/noisy categories — `title_status` kept only clean/rebuilt/lien; removed 'new' and 'salvage' conditions; removed electric/hybrid/other fuel; removed bus/offroad/other types; removed 'other' transmission
- **Feature engineering:** Created `vehicle_age` (2026 − year); converted `cylinders` from string categories to numeric integers
- **Encoding:** `OneHotEncoder` for all categorical features
- **Scaling:** `StandardScaler` on numeric features (`odometer`, `cylinders`, `vehicle_age`)
- **Target transformation:** `TransformedTargetRegressor` with `np.log1p` / `np.expm1` applied to all models
- **Train/test split:** 80/20 split with `random_state=42`

### 4. Modeling

Three model types were trained on the top 10 features (transmission dropped due to negative permutation importance):

| Model | Approach | Test R² | Test MAE |
|---|---|---|---|
| **Ridge Regression** | Linear, log target, GridSearchCV alpha tuning | 0.676 | $4,584 |
| **Lasso Regression** | Linear, log target, GridSearchCV alpha tuning | 0.674 | $4,594 |
| **Random Forest** | Non-linear ensemble, log target | **0.829** | **$2,988** |

All models used:
- `run_model()` utility for consistent pipeline construction, fitting, and evaluation
- `Pipeline` with `ColumnTransformer` for preprocessing
- `TransformedTargetRegressor` with log1p/expm1 target transformation
- `compare_models()` utility for dynamic side-by-side comparison (no hardcoded metrics)
- `permutation_importance` (n_repeats=10, evaluated on test set) for feature ranking

### 5. Evaluation

All three models achieve **positive R²** on the held-out test set. The best model (Random Forest) explains ~83% of price variance with an average error of ~$3,000 per vehicle. Ridge and Lasso are simpler and more interpretable but explain ~68%.

**Permutation importance analysis** ranked features by their contribution:

| Rank | Feature | Mean Importance | Role |
|---|---|---|---|
| 1 | **vehicle_age** | 64,532,709 | Strongest predictor — newer cars are worth more |
| 2 | **odometer** | 39,923,929 | Second strongest — lower mileage = higher price |
| 3 | **type** | 29,099,878 | Trucks/SUVs hold value better than sedans |
| 4 | **fuel** | 14,423,952 | Diesel holds value in truck/towing markets |
| 5 | **drive** | 8,084,586 | 4WD/AWD commands a premium over FWD |
| 6 | **condition** | 5,990,368 | Excellent/like-new fetches higher prices |
| 7 | **manufacturer** | 5,839,261 | Brand premiums (Toyota, Honda, luxury brands) |
| 8 | **cylinders** | 4,405,513 | Larger engines correlate with higher value |
| 9 | **title_status** | 1,053,523 | Clean titles worth more (small effect after filtering) |
| 10 | **paint_color** | 586,441 | Minimal effect on pricing |
| 11 | **transmission** | −2,133,214 | Negative importance — dropped from final models |

### 6. Deployment

Findings are delivered as a concise client-facing report within the notebook, written for an audience of used car dealers.

## Key Findings & Recommendations for the Dealership

### Top Price Drivers

| Rank | Factor | Recommendation |
|---|---|---|
| 1 | **Vehicle Age** | Prioritize newer inventory (under 10 years old). Each additional year significantly reduces value. |
| 2 | **Mileage (Odometer)** | Seek low-mileage acquisitions. Highlight low mileage in listings. |
| 3 | **Vehicle Type** | Lean into trucks and SUVs — they hold value better than sedans. |
| 4 | **Fuel Type** | Diesel vehicles retain value, especially in truck/towing markets. |
| 5 | **Drive Type** | 4WD/AWD commands a premium over FWD. |
| 6 | **Condition** | Invest in reconditioning — moving "fair" to "good" yields price uplift. |
| 7 | **Manufacturer** | Stock brands with strong resale value (Toyota, Honda, luxury brands). |

### Factors with Minimal Impact

- **Paint color** — barely affects resale value
- **Transmission** — not a meaningful differentiator; dropped from final models

## Technologies Used

- **Python 3.7**
- **pandas** — data manipulation
- **NumPy** — numerical operations
- **matplotlib / seaborn** — visualization
- **scikit-learn** — modeling (`Ridge`, `Lasso`, `LinearRegression`, `RandomForestRegressor`, `GridSearchCV`, `Pipeline`, `ColumnTransformer`, `TransformedTargetRegressor`, `OneHotEncoder`, `StandardScaler`, `PolynomialFeatures`, `permutation_importance`)

## How to Run

1. Ensure you have Python 3.7+ and the required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
   ```
2. Open and run `prompt_II.ipynb` in Jupyter Notebook or VS Code.
3. The notebook reads data from `data/vehicles.csv` — ensure this file is present.
