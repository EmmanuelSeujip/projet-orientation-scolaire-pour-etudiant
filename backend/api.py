from fastapi import FastAPI

app = FastAPI()
@app.get("/")
def great():
    return {"message": "Hello World"}