from pydantic import BaseModel


class News(BaseModel):
    msg: str
