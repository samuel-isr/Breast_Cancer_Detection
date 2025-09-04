# **Breast Cancer Detection using AI**

A specialized AI system for breast cancer, utilizing **tabular data** with **98.25% test accuracy**, providing AI-powered analysis and foundational Explainable AI (XAI) insights.

# **The AI Brain**
At the heart of this application is a custom-built deep learning model. Think of it as a highly specialized brain that has been trained for one specific task: to distinguish between benign and malignant cells based on their measurements. This was achieved through a process called Supervised Learning.

*Learning from Examples:* 
The model was trained on the Breast Cancer Wisconsin (Diagnostic) dataset, which contains hundreds of examples, each already labeled by medical experts. By studying these examples, the model learned to identify the subtle patterns and combinations of features that are characteristic of each diagnosis.

# *The Neural Network:* 
The model's architecture is a Neural Network, inspired by the structure of the human brain. It's composed of layers of interconnected nodes, or "neurons." The input layer receives the 30 feature measurements. This information then travels through several hidden layers, where complex relationships between the features are analyzed. Finally, the output layer produces a single probability—the model's confidence that the sample is malignant.

*Achieving High Accuracy:* 
The model was built using the popular TensorFlow and Keras libraries. During its training, it continuously refined its predictions, aiming to minimize errors. This iterative process allowed it to achieve a very high accuracy (over 98%) on data it had never seen before, making it a powerful tool for this diagnostic task.


# **Explainable AI (XAI) in Action**

A major focus of this project is moving beyond a "black box" prediction.
This application implements Explainable AI (XAI) to provide insight into the model's reasoning. We use the SHAP (SHapley Additive exPlanations) technique, a leading method for explaining the output of any machine learning model.

When you make a prediction, the SHAP algorithm analyzes how much each individual feature (like 'concave points_worst' or 'texture_mean') contributed to the final decision. The feature importance chart visualizes this, showing you the top factors that pushed the model's prediction one way or the other. This transparency is crucial for building trust and understanding in AI-driven diagnostic tools.


 # Usage Setup and Installation
 
To run this project locally, please follow these steps:
1. Clone the Repository
git clone
cd your-repo-name

2. Create and Activate a Virtual Environment It's highly recommended to use a virtual environment to keep project dependencies isolated.
Create the virtual environment:
python -m venv venv

Activate it (Windows):
.\venv\Scripts\activate

Activate it (macOS/Linux):
source venv/bin/activate

3. Install Dependencies Install all the necessary Python packages using the requirements.txt file.
pip install -r requirements.txt

The application consists of two parts that need to be run simultaneously: the backend server and the frontend interface.

1. Start the Backend Server Open a terminal, activate your virtual environment, and run the Flask app:
   
python app.py

You should see output indicating that the server is running on http:// (example).
Keep this terminal running.

2. Launch the Frontend Open the index.html file in your web browser.
The easiest way to do this with live reloading is by using the Live Server extension in VS Code (right-click index.html -> "Open with Live Server").

You can now interact with the application in your browser.
