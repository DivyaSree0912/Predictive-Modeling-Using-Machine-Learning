# 💻 Laptop Price Prediction Using Machine Learning

> A machine learning project that predicts laptop prices based on hardware specifications and analyzes the factors that influence laptop pricing.

## 📌 Overview

**Laptop Price Prediction** is a machine learning regression project that analyzes laptop specifications and predicts their prices.

The project uses a dataset containing **1,303 laptop records** with information such as:

* Brand
* Laptop type
* RAM
* CPU
* GPU
* Storage
* Operating system
* Screen specifications
* Weight
* Price

The project follows a complete machine learning workflow:

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Data Preprocessing
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Save Trained Model
```

Two regression algorithms are implemented and compared:

* Linear Regression
* Random Forest Regressor

The **Random Forest Regressor** achieved the better result on the test set used in the notebook.

---

# 🎯 Objectives

The main objectives of this project are to:

* Analyze laptop specifications and pricing patterns
* Understand which hardware features influence laptop prices
* Clean and preprocess real-world laptop data
* Perform exploratory data analysis
* Engineer useful features from raw specifications
* Train regression models for price prediction
* Compare different machine learning algorithms
* Evaluate model performance using regression metrics
* Save the trained model for potential future use

---

# 📊 Dataset

The dataset used in this project is a laptop pricing dataset obtained from **Kaggle**.

The original dataset contains **1,303 rows and 13 columns**.

### Dataset Features

| Feature            | Description                             |
| ------------------ | --------------------------------------- |
| `laptop_ID`        | Unique laptop identifier                |
| `Company`          | Laptop manufacturer                     |
| `Product`          | Product/model name                      |
| `TypeName`         | Type/category of laptop                 |
| `Inches`           | Screen size                             |
| `ScreenResolution` | Display resolution and display features |
| `Cpu`              | Processor information                   |
| `Ram`              | RAM capacity                            |
| `Memory`           | Storage configuration                   |
| `Gpu`              | Graphics processor                      |
| `OpSys`            | Operating system                        |
| `Weight`           | Laptop weight                           |
| `Price_euros`      | Laptop price in euros — target variable |

### Target Variable

```text
Price_euros
```

The goal is to predict the price of a laptop based on its specifications.

---

# 🧹 Data Preprocessing

The project performs several preprocessing operations before model training.

## Duplicate Removal

Duplicate records are checked and removed:

```python
df = df.drop_duplicates()
```

## RAM Conversion

RAM values such as:

```text
8GB
16GB
32GB
```

are converted into numerical values.

Example:

```text
8GB → 8
16GB → 16
```

## Weight Conversion

Weight values such as:

```text
1.37kg
2.5kg
```

are converted into numerical values.

## Unnecessary Columns

The following columns are removed before model training:

```text
laptop_ID
Product
ScreenResolution
Cpu
Memory
```

The remaining features are used for prediction after preprocessing.

---

# 🧠 Feature Engineering

Feature engineering is performed to extract useful information from the original dataset.

## Touchscreen

A new binary feature is created:

```text
TouchScreen
```

It identifies whether a laptop has touchscreen functionality.

```python
df['TouchScreen'] = df['ScreenResolution'].apply(
    lambda x: 1 if 'Touchscreen' in x else 0
)
```

Values:

```text
1 → Touchscreen
0 → No Touchscreen
```

---

## IPS Display

A new feature called:

```text
Ips
```

is created based on the screen-resolution information.

```text
1 → IPS display
0 → Non-IPS display
```

---

## SSD

The project extracts whether the laptop contains SSD storage:

```text
SSD = 1 → SSD present
SSD = 0 → SSD not detected
```

---

## HDD

Similarly, an HDD indicator is created:

```text
HDD = 1 → HDD present
HDD = 0 → HDD not detected
```

---

## CPU Brand

The original CPU descriptions are simplified into broader categories:

```text
Intel Core i7
Intel Core i5
Intel Core i3
AMD Processor
Other Intel Processor
```

This produces a new feature:

```text
Cpu_Brand
```

This allows the models to work with a more generalized representation of processor information.

---

# 📈 Exploratory Data Analysis

Several visualizations are created to understand relationships between laptop specifications and price.

### Analyses performed include:

* Price distribution
* Company vs. price
* RAM vs. price
* Touchscreen vs. price
* IPS display vs. price
* SSD vs. price
* HDD vs. price
* CPU category vs. price

### Example questions explored

```text
Does more RAM correspond to higher prices?

Do laptops with SSD storage tend to cost more?

Does an IPS display affect price?

Does touchscreen functionality influence price?

How does processor category relate to price?

How does laptop brand affect price?
```

The notebook also includes:

* Actual vs. predicted price visualization
* Prediction-error distribution

---

# 🔢 Data Encoding

Categorical variables are converted into numerical representations using **One-Hot Encoding**.

```python
X = pd.get_dummies(X, drop_first=True)
```

This allows categorical variables such as:

```text
Company
TypeName
Gpu
OpSys
Cpu_Brand
```

to be used by the machine learning algorithms.

---

# 🤖 Machine Learning Models

Two regression models are trained and evaluated.

## 1. Linear Regression

Linear Regression is used as a baseline model.

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, y_train)
```

### Test Performance

| Metric   | Score |
| -------- | ----: |
| R² Score | ~0.74 |
| MAE      |  ~242 |

---

## 2. Random Forest Regressor

A Random Forest Regressor is used as the second model.

Configuration:

```python
RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
```

### Test Performance

| Metric   | Score |
| -------- | ----: |
| R² Score | ~0.78 |
| MAE      |  ~197 |

---

# 📊 Model Comparison

| Model                   | R² Score |  MAE |
| ----------------------- | -------: | ---: |
| Linear Regression       |    ~0.74 | ~242 |
| Random Forest Regressor |    ~0.78 | ~197 |

### Interpretation

The Random Forest model produced:

* A higher R² score
* A lower Mean Absolute Error

on the test split used in the notebook.

Therefore, the Random Forest model was selected as the trained model and saved for future use.

> **Note:** These values correspond to the evaluation performed in the project notebook using the specified train/test split and preprocessing pipeline.

---

# 📉 Model Evaluation

## R² Score

R² measures how much of the variation in the target variable is explained by the model.

A higher R² generally indicates that the model explains more of the observed variation in the target data.

---

## Mean Absolute Error

MAE measures the average absolute difference between actual and predicted prices.

A lower MAE indicates smaller average prediction errors.

For this project, the Random Forest model achieved an MAE of approximately:

```text
197
```

meaning the average absolute prediction error on the evaluated test set was approximately **€197**.

---

# 📈 Actual vs Predicted Prices

The notebook generates a scatter plot comparing:

```text
Actual Price
        vs.
Predicted Price
```

This helps visually evaluate how closely the model's predictions follow the actual laptop prices.

---

# 📊 Prediction Error Analysis

The project also calculates prediction errors:

```python
errors = y_test - y_pred_rf
```

and visualizes their distribution using a histogram.

This helps analyze how the model's prediction errors are distributed across the test dataset.

---

# 💾 Trained Model

After training, the Random Forest model is serialized using Python's `pickle` module.

```python
import pickle

pickle.dump(
    rf,
    open('laptop_price_model.pkl', 'wb')
)
```

The resulting model file is:

```text
laptop_price_model.pkl
```

This allows the trained model to be reused without retraining it every time.

---

# 🛠️ Tech Stack

| Technology       | Purpose                      |
| ---------------- | ---------------------------- |
| Python           | Programming language         |
| Pandas           | Data manipulation            |
| NumPy            | Numerical operations         |
| Matplotlib       | Data visualization           |
| Seaborn          | Statistical visualization    |
| Scikit-learn     | Machine learning             |
| Pickle           | Model serialization          |
| Google Colab     | Development environment      |
| Jupyter Notebook | Experimentation and workflow |

---

# 📁 Project Structure

```text
Predictive-Modeling-Using-Machine-Learning/
│
├── Laptop_price_prediction.ipynb
│
├── laptop_data.csv
│
├── laptop_price_model.pkl
│
└── OVERVIEW.md
```

### `Laptop_price_prediction.ipynb`

Contains the complete machine learning workflow:

* Dataset loading
* Data inspection
* Data cleaning
* Exploratory data analysis
* Feature engineering
* Preprocessing
* Model training
* Model evaluation
* Visualization
* Model serialization

### `laptop_data.csv`

Contains the original laptop specification and pricing data.

### `laptop_price_model.pkl`

Contains the trained Random Forest regression model.

### `OVERVIEW.md`

Contains the original project overview and model results.

---

# 🚀 How to Run

## Option 1 — Google Colab

The notebook was developed using Google Colab.

1. Open `Laptop_price_prediction.ipynb` in Google Colab.
2. Upload `laptop_data.csv`.
3. Run the notebook cells sequentially.
4. The notebook performs preprocessing, visualization, training, and evaluation.
5. The trained model is saved as `laptop_price_model.pkl`.

---

## Option 2 — Local Jupyter Environment

Clone the repository:

```bash
git clone https://github.com/DivyaSree0912/Predictive-Modeling-Using-Machine-Learning.git
```

Move into the project directory:

```bash
cd Predictive-Modeling-Using-Machine-Learning
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
Laptop_price_prediction.ipynb
```

and execute the cells sequentially.

---

# 🔄 Machine Learning Pipeline

```text
             Laptop Dataset
                    │
                    ▼
          Data Inspection
                    │
                    ▼
          Duplicate Removal
                    │
                    ▼
          Feature Engineering
          ┌─────────┼─────────┐
          │         │         │
       Touch      IPS       Storage
       Screen     Display   Features
          │         │         │
          └─────────┼─────────┘
                    │
                    ▼
             CPU Categorization
                    │
                    ▼
            Categorical Encoding
                    │
                    ▼
              Train / Test Split
                    │
             ┌──────┴──────┐
             ▼             ▼
      Linear Regression  Random Forest
             │             │
             └──────┬──────┘
                    ▼
             Model Evaluation
                    │
                    ▼
              Model Comparison
                    │
                    ▼
         Save Random Forest Model
```

---

# 🔍 Key Insights

The exploratory analysis provides several useful observations from the dataset:

* Laptop prices vary considerably across manufacturers and product categories.
* RAM capacity shows a relationship with laptop pricing.
* SSD-equipped laptops generally appear at higher price points in the dataset.
* Processor category is associated with price differences.
* Display features such as touchscreen and IPS panels can contribute to price differences.
* Hardware specifications collectively provide useful information for predicting laptop prices.

These observations describe patterns found in the dataset and should not be interpreted as universal pricing rules for every laptop market.

---

# ⚠️ Limitations

The current project has several limitations:

* The dataset represents a particular collection of laptop listings and may not reflect current market prices.
* Prices are represented in euros.
* The model is trained on historical/static data.
* No live e-commerce or market-price API is connected.
* The current notebook does not provide a deployed prediction API or web interface.
* Some original features are removed during preprocessing.
* The model is evaluated using a single train/test split.
* The feature engineering approach uses simplified rules for CPU and storage categories.

---

# 🔮 Future Enhancements

Possible improvements include:

### 🌐 Web Application

Build an interactive web interface where users can enter laptop specifications and receive an estimated price.

### 🔌 Prediction API

Deploy the trained model using:

* FastAPI
* Flask

and expose a prediction endpoint.

### 📱 User Interface

Create a frontend using:

* React
* HTML/CSS/JavaScript

for interactive laptop price prediction.

### 📊 More Advanced Models

Experiment with:

* Gradient Boosting
* XGBoost
* LightGBM
* CatBoost

and compare their performance.

### ⚙️ Hyperparameter Optimization

Use techniques such as:

* GridSearchCV
* RandomizedSearchCV
* Cross-validation

to optimize model parameters.

### 📈 Feature Importance

Analyze Random Forest feature importance to identify which laptop specifications contribute most strongly to predictions.

### 🔄 Updated Dataset

Train the model using newer laptop prices and specifications to improve relevance to current markets.

### ☁️ Deployment

Deploy the prediction system using a cloud platform and expose it as an accessible machine learning application.

---

# 📌 Project Highlights

* Real-world laptop pricing dataset
* 1,303 laptop records
* Data cleaning and preprocessing
* Exploratory data analysis
* Feature engineering
* One-Hot Encoding
* Multiple regression algorithms
* Model comparison
* R² and MAE evaluation
* Actual vs. predicted analysis
* Prediction-error analysis
* Trained Random Forest model saved with Pickle

---

# 👩‍💻 Author

**Dadi Divya Sree**

GitHub:
https://github.com/DivyaSree0912

---

