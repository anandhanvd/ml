from student_level_predictor import StudentLevelPredictor
import os

def main():
    predictor = StudentLevelPredictor()
    
    # Train model and print evaluation metrics
    print("Training model...")
    metrics = predictor.train()
    
    print("\nModel Performance Metrics:")
    print(f"Training Score: {metrics['train_score']:.4f}")
    print(f"Test Score: {metrics['test_score']:.4f}")
    
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Test prediction
    test_features = {
        'time_per_question': 2.5,
        'question_difficulty': 7.0,
        'topic_difficulty': 6.0,
        'score_percentage': 85.0
    }
    
    prediction = predictor.predict(test_features)
    print("\nTest Prediction:")
    print(f"Predicted Level: {prediction['predicted_level']}")
    print("Confidence Scores:", prediction['confidence_scores'])

if __name__ == "__main__":
    main() 