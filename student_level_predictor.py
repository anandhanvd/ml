import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

class StudentLevelPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
    def train(self, data_path='training_data.csv'):
        # Load and prepare data
        df = pd.read_csv(data_path)
        
        # Prepare features and target
        X = df[['time_per_question', 'question_difficulty', 
               'topic_difficulty', 'score_percentage']]
        y = df['recommended_level']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Fit model
        self.model.fit(X_train_scaled, y_train)
        
        # Save model and scaler
        joblib.dump(self.model, 'models/student_level_model.joblib')
        joblib.dump(self.scaler, 'models/feature_scaler.joblib')
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Generate predictions and report
        y_pred = self.model.predict(X_test_scaled)
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def predict(self, features):
        """
        Predict student level based on quiz performance
        
        Args:
            features (dict): {
                'time_per_question': float,
                'question_difficulty': float,
                'topic_difficulty': float,
                'score_percentage': float
            }
        """
        if self.model is None:
            self.load_model()
            
        feature_array = np.array([[
            features['time_per_question'],
            features['question_difficulty'],
            features['topic_difficulty'],
            features['score_percentage']
        ]])
        
        scaled_features = self.scaler.transform(feature_array)
        prediction = self.model.predict(scaled_features)[0]
        
        # Simplified response to match existing frontend
        return {
            'level': prediction,  # Just return the predicted level without confidence scores
            'message': f'Based on your performance, you are at {prediction} level.'
        }
    
    def load_model(self):
        """Load saved model and scaler"""
        try:
            self.model = joblib.load('models/student_level_model.joblib')
            self.scaler = joblib.load('models/feature_scaler.joblib')
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}") 