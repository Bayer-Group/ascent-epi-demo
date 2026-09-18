from pydantic import BaseModel


class SQLToQuestionVerdict(BaseModel):
    """
    Represents a model for SQL-to-Question verdict assessment.

    This class is used to evaluate the similarity of SQL queries to their
    corresponding questions and provide an explanation for the verdict.
    """
    similar: bool
    explanation: str
