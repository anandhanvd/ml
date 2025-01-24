from student_level_predictor import StudentLevelPredictor
import os

def main():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    predictor = StudentLevelPredictor()
    print("Training model...")
    metrics = predictor.train('training_data.csv')  # Note: path relative to current directory
    
    print("\nModel Performance Metrics:")
    print(f"Training Score: {metrics['train_score']:.4f}")
    print(f"Test Score: {metrics['test_score']:.4f}")

if __name__ == "__main__":
    main() 