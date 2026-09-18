from pydantic import BaseModel


class ReformatSQL(BaseModel):
    sql: str
    success: bool
