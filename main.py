from fastapi import FastAPI

app = FastAPI(title="Incident War Room")


@app.get("/")
def home():
    return {
        "message": "Incident War Room is running!",
        "status": "healthy",
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}
