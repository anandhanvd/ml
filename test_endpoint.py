import requests

def test_prediction():
    url = "http://127.0.0.1:8000/mcq/predict-level"
    
    # Test data matching the frontend format
    test_data = {
        "score": 6,
        "time_taken": 42
    }
    
    print("Sending request with data:", test_data)
    try:
        response = requests.post(url, json=test_data)
        print("\nStatus Code:", response.status_code)
        print("Response:", response.text)
        
        if response.status_code == 200:
            print("\nSuccess! The prediction system is working.")
        else:
            print("\nError: Something went wrong with the request.")
            
    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    test_prediction() 