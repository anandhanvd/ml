import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from enum import Enum
import google.generativeai as genai
from dotenv import load_dotenv
import json
import uvicorn
import re
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
import asyncio
from student_level_predictor import StudentLevelPredictor

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_GEMINI_KEY"))

generation_config = {
    "temperature": 0.3,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change to specific origins as needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class UserLevel(str, Enum):
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"

# Initialize predictor and train model if needed
predictor = StudentLevelPredictor()
try:
    predictor.load_model()
except Exception as e:
    print("Training new model...")
    predictor.train()

class QuizResult(BaseModel):
    score: float = Field(..., description="Quiz score")
    time_taken: float = Field(..., description="Time taken in seconds")


def predict_user_level(score: float, time_taken: float) -> UserLevel:
    """
    Predict user level based on quiz score and time taken.
    """
    if score >= 7 and time_taken <= 80:
        return UserLevel.ADVANCED
    elif 4 <= score < 7:
        return UserLevel.INTERMEDIATE
    else:
        return UserLevel.BEGINNER
    

@app.post("/predict-level")
async def predict_level(quiz_result: QuizResult):
    """
    Predict the user's level based on quiz score and time taken
    """
    try:
        level = predict_user_level(quiz_result.score, quiz_result.time_taken)
        
        # Return the level in lowercase to match frontend expectations
        return level.value.lower()
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error making prediction: {str(e)}"
        )

class CourseRequest(BaseModel):
    subject: str = Field(..., description="The subject of the course")
    difficulty: DifficultyLevel = Field(..., description="Difficulty level of the course")
    focus_area: str = Field(..., description="Specific area to focus on within the subject")
    units: int = Field(..., ge=1, le=10, description="Number of units desired")



async def generate_mcqs(unit_data: dict, subject: str, difficulty: str, focus_area: str):
    """Generate MCQs for the unit"""
    mcq_prompt = f"""
    Generate Multiple Choice Questions (MCQs) specifically for:
    Subject: {subject}
    Focus Area: {focus_area}
    Unit Title: {unit_data['unitTitle']}
    Difficulty: {difficulty}

    Important Instructions:
    1. Questions MUST be directly related to {subject} and {focus_area}
    2. All questions should match the {difficulty} difficulty level
    3. Each question should test understanding of {unit_data['unitTitle']}
    4. Include practical applications and real-world scenarios
    5. Ensure explanations are clear and educational
    6. there can be calculations in the questions but the answer should be a whole number

    Return the response in this JSON format:
    {{
        "assessment": {{
            "unitAssessment": [
                {{
                    "topic": "{subject} - {focus_area}",
                    "questions": [
                        {{
                            "questionId": "unique_id",
                            "question": "Question text",
                            "options": ["Option A", "Option B", "Option C", "Option D"],
                            "correctAnswer": "Correct option",
                            "explanation": "Detailed explanation"
                        }}
                    ]
                }}
            ]
        }}
    }}

    Generate exactly 3 MCQs that are highly relevant to {subject} and {focus_area}.
    """
    
    try:
        response = model.generate_content(mcq_prompt)
        cleaned_json = re.sub(r"^```json|```$", "", response.text, flags=re.MULTILINE).strip()
        mcq_data = json.loads(cleaned_json)
        
        # Validate that questions are relevant
        for topic in mcq_data["assessment"]["unitAssessment"]:
            if not any(focus_area.lower() in q["question"].lower() or 
                      subject.lower() in q["question"].lower() 
                      for q in topic["questions"]):
                raise ValueError("Generated questions are not relevant to the subject and focus area")
        
        return mcq_data
    except Exception as e:
        print(f"Error generating MCQs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate relevant MCQs: {str(e)}"
        )

async def get_unit_details(unit_title: str, subject: str, difficulty: str, focus_area: str):
    """Generate unit structure and MCQs"""
    unit_prompt = f"""
    Generate a detailed unit structure for "{unit_title}" in {subject} course.
    Difficulty level: {difficulty}
    Focus area: {focus_area}

    Return the response in this JSON format:
    {{
        "unitTitle": "{unit_title}"
    }}

    Ensure content matches the difficulty level and focuses on practical applications.
    """
    
    try:
        response = model.generate_content(unit_prompt)
        cleaned_json = re.sub(r"^```json|```$", "", response.text, flags=re.MULTILINE).strip()
        print(f"Unit structure response for {unit_title}: {cleaned_json}")
        
        unit_data = json.loads(cleaned_json)
        
        # Generate MCQs for the unit
        unit_mcqs = await generate_mcqs(unit_data, subject, difficulty, focus_area)
        
        # Merge the unit structure with MCQs
        unit_data["assessment"] = unit_mcqs
        
        return unit_data
        
    except Exception as e:
        print(f"Error in get_unit_details for {unit_title}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate unit details for {unit_title}: {str(e)}"
        )

@app.post("/generate-question")
async def generate_course(request: CourseRequest):
    """Generate a complete course with MCQs for each unit"""
    try:
        # Generate course structure
        structure_prompt = f"""
        Generate a comprehensive course structure for {request.subject} with exactly {request.units} units.
        Focus area: {request.focus_area}
        Difficulty: {request.difficulty}

        Return ONLY unit titles in this JSON format:
        {{
            "courseTitle": "",
            "difficultyLevel": "",
            "description": "",
            "prerequisites": ["prerequisite 1", "prerequisite 2"],
            "learningOutcomes": ["outcome 1", "outcome 2"],
            "units": [
                {{
                    "unitTitle": "",
                    "unitDescription": ""
                }}
            ],
            "overview": "",
            "assessmentMethods": ["method 1", "method 2"]
        }}
        """
        
        structure_response = model.generate_content(structure_prompt)
        cleaned_json = re.sub(r"^```json|```$", "", structure_response.text, flags=re.MULTILINE).strip()
        print(f"Course structure response: {cleaned_json}")
        
        course_structure = json.loads(cleaned_json)

        # Concurrently generate unit details
        unit_tasks = [
            get_unit_details(
                unit["unitTitle"],
                request.subject,
                request.difficulty,
                request.focus_area
            ) for unit in course_structure["units"]
        ]
        
        detailed_units = await asyncio.gather(*unit_tasks, return_exceptions=True)
        
        # Filter out failed units
        course_structure["units"] = [
            unit for unit in detailed_units if not isinstance(unit, Exception)
        ]
        
        if not course_structure["units"]:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate any unit details"
            )
        
        return course_structure
        
    except Exception as e:
        print(f"Error in generate_course: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=7000)