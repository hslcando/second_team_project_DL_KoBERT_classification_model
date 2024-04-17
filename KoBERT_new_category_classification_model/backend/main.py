from fastapi import FastAPI
from api import news_classifier

app = FastAPI()
app.include_router(news_classifier.router)


@app.get("/")
def haelth_check_handler():
    return {"message": "Hello World!"}
