# Iris-Flower-Classification

📌 Project Overview
This project implements an Iris Flower Classification Model using machine learning. It trains a model to classify iris flowers into three species (Setosa, Versicolor, and Virginica) based on four features:
Sepal Length
Sepal Width
Petal Length
Petal Width

🚀 Technologies Used
Python
NumPy
Pandas
Scikit-Learn
Matplotlib & Seaborn
Joblib (for model serialization)

📂 Project Structure
Iris-Flower-Classification/
│── README.md              # Project description
│── iris_classification.ipynb  # Jupyter Notebook with ML code
│── iris_dataset.csv       # Dataset file
│── model.pkl              # Trained ML model
│── requirements.txt       # Dependencies file
│── images/                # Folder with plots & figures
│   ├── confusion_matrix.png
│   ├── heatmap.png
│   ├── accuracy_plot.png

🛠 Installation & Setup

🔹 Step 1: Clone the Repository
git clone https://github.com/Riyapandey389/Iris-Flower-Classification.git
cd Iris-Flower-Classification

🔹 Step 2: Create a Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate   # For Mac/Linux
venv\Scripts\activate      # For Windows

🔹 Step 3: Install Dependencies
pip install -r requirements.txt


📊 Dataset Description (famous Iris.csv Dataset)

The dataset consists of 150 samples with 4 features and 1 target variable (species type).


📊 Model Training & Evaluation

🔹 Step 1: Data Preprocessing
Loaded dataset using Pandas
Performed Exploratory Data Analysis (EDA)
Visualized feature correlations using heatmaps

🔹 Step 2: Model Training
Split data into training (80%) and testing (20%)
Used Random Forest Classifier for training

🔹 Step 3: Model Evaluation
Achieved 100% accuracy on test data
Used confusion matrix & classification report for validation


🎯 How to Use the Pretrained Model (model.pkl)?
You can load and use the pretrained model without retraining:

import joblib

# Load the saved model
model = joblib.load('model.pkl')

# Example input: Sepal Length, Sepal Width, Petal Length, Petal Width
sample_input = [[5.1, 3.5, 1.4, 0.2]]

# Predict the flower species
prediction = model.predict(sample_input)
print("Predicted Class:", prediction)


📸 Visualizations

🔹 Confusion Matrix

🔹 Feature Correlation Heatmap


🤖 Future Improvements

Implement Hyperparameter Tuning to further optimize performance
Deploy the model using Flask/FastAPI for real-time predictions
Build a Web UI to allow users to input flower dimensions and get predictions


📜 License

This project is open-source and available under the MIT License.

🤝 Contributing

Want to contribute? Follow these steps:
1. Fork the repository
2. Create a new branch (feature-branch)
3. Commit your changes
4. Submit a Pull Request


📬 Contact
For any queries or suggestions, reach out to:
📧 Email: tech.riya005@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/riya-pandey-8b8454290
