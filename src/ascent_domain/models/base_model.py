from pydantic import BaseModel, ConfigDict


class AscentBaseModel(BaseModel):
    model_config = ConfigDict(validate_assignment=True, from_attributes=True, extra="ignore", populate_by_name=True)
