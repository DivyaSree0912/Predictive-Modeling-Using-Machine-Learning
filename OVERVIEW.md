Laptop Price Prediction Using Machine Learning

Project Overview
This project predicts laptop prices based on laptop specifications using Machine Learning.

The project helps understand how features like:
- RAM
- Processor
- SSD/HDD
- Brand
- Weight
affect laptop prices.


Dataset
Dataset taken from Kaggle.

Dataset contains:
- Company
- Laptop Type
- RAM
- CPU
- GPU
- Memory
- Weight
- Operating System
- Price


Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab


Project Steps

 1. Data Cleaning
- Checked missing values
- Checked duplicate values
- Cleaned dataset

 2. Exploratory Data Analysis
Created graphs to understand:
- Company vs Price
- RAM vs Price
- SSD vs Price
- CPU vs Price
- Touchscreen vs Price

 3. Feature Engineering
Created new features:
- TouchScreen
- IPS Display
- SSD
- HDD
- CPU Brand

 4. Data Preprocessing
Converted text data into numerical format using One-Hot Encoding.

 5. Model Training
Used:
- Linear Regression
- Random Forest Regressor

 6. Model Evaluation
Compared model performance using:
- R2 Score
- Mean Absolute Error (MAE)


Model Results

 Linear Regression
- R2 Score: 0.74
- MAE: 242

 Random Forest Regressor
- R2 Score: 0.78
- MAE: 197

Random Forest performed better than Linear Regression.


Conclusion
This project successfully predicts laptop prices using Machine Learning.

From the analysis:
- Higher RAM increases laptop price
- SSD laptops are more expensive
- Intel Core i7 laptops are costly
- Touchscreen and IPS displays increase price

Random Forest gave better prediction accuracy.



# Future Improvements
- Build Streamlit Web App
- Deploy project online
- Improve accuracy using advanced models
- Add user input prediction system

---

# Author
Divya Sree
