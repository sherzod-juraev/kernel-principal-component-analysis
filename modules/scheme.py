from pydantic import BaseModel, field_validator
from numpy import array


class KernelPCAOut(BaseModel):
    pass


class KernelPCAIn(BaseModel):
    pass