from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import your existing apps
from fullmcqgen import app as mcq_app
from courserecommendataion import app as course_app
from fullcoursegen import app as generator_app
from contentlabelall import app as content_app

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all your existing apps as sub-applications
app.mount("/mcq", mcq_app)
app.mount("/course", course_app)
app.mount("/generator", generator_app)
app.mount("/content", content_app)

@app.get("/")
async def root():
    return {"message": "Education ML API is running"}