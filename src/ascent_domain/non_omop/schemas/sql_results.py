from typing import Any, List

import pandas as pd
from pydantic import BaseModel, Field


class Frame(BaseModel):
    """
    A serializable representation of tabular data, compatible with pandas DataFrames.

    This class provides a way to store and transfer tabular data in a format that
    can be easily serialized to JSON and deserialized back, while maintaining
    compatibility with pandas DataFrames.
    """

    columns: List[str]
    index: List[Any] = Field(default_factory=list)
    data: List[List[Any]] = Field(default_factory=list)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> "Frame":
        """
        Convert a pandas DataFrame to a Frame object.

        Args:
            df (pd.DataFrame): The pandas DataFrame to convert

        Returns:
            Frame: A Frame object representing the same data as the DataFrame
        """
        return cls.model_validate(df.to_dict(orient="split"))

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert this Frame object to a pandas DataFrame.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the same data as this Frame
        """
        return pd.DataFrame(self.data, columns=self.columns, index=self.index)


class LLMOutputSQLHealing(BaseModel):
    """
    Model for the output of SQL healing by an LLM.

    This class represents the response from an LLM when asked to heal
    (fix/correct) an invalid SQL query. It contains the corrected SQL query
    that should resolve the errors in the original query.
    """

    healed_query: str
