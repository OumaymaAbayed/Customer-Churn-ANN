# 🧠 Artificial Neural Network for Customer Churn Prediction
## 📄 Project Overview
This project builds and trains an Artificial Neural Network (ANN) to predict customer churn based on the Churn_Modelling.csv dataset.
It uses deep learning techniques to classify whether a customer is likely to leave a bank based on various features like credit score, geography, gender, age, balance, and more.

## 🛠️ Technologies Used
- Python 3.x
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib / Seaborn (for optional visualization)

## 📚 Dataset
**Dataset:** '''Churn_Modelling.csv'''
**Source:** Kaggle
- Rows: 10,000 customer records
- Columns: 14 features including:

    Credit Score
    
    Geography
    
    Gender
    
    Age
    
    Tenure
    
    Balance
    
    Number of Products
    
    Has Credit Card
    
    Is Active Member
    
    Estimated Salary
    
    Exited (Target Variable)

## ⚙️ Project Structure
'''
├── scripts
  ├── ANN.ipynb           # Main Jupyter Notebook containing all code
├── data  
  ├── Churn_Modelling.csv # Dataset
├── README.md           # Project documentation (you are reading it)
'''

## 🔥 Main Steps in the Project
### Data Preprocessing:

- Load data
- Encode categorical variables (Label Encoding and One-Hot Encoding)
- Split data into training and testing sets
- Feature Scaling (StandardScaler)

### Build the ANN Model:

- Input layer with proper shape
- Hidden layers with ReLU activation
- Output layer with Sigmoid activation (binary classification)

### Train the Model:

- Compile with adam optimizer and binary_crossentropy loss
- Train over multiple epochs

### Evaluate the Model:

- Calculate test loss and test accuracy
- Interpret result
- Suggest improvements

## 📈 Results
**Test Accuracy:** ~84.45%

**Test Loss:** ~0.689

The model achieves a decent performance but could be improved with hyperparameter tuning, more data augmentation, and architecture enhancements.

## 💡 Possible Improvements
- Use Cross-Validation for better evaluation
- Tune Hyperparameters (learning rate, batch size, number of layers, neurons)
- Add Regularization (Dropout, L2 Regularization)
- Handle class imbalance if needed
- Try different model architectures (e.g., deeper networks, CNNs on structured data)

##✍️ Author
Oumayma Abayed

##📜 Credit
https://www.kaggle.com/code/youssefmagdy131/ann-implementation
