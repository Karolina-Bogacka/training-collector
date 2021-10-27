from typing import Optional

from pydantic import BaseModel


class TrainingConfiguration(BaseModel):
    model: str
    epochs: int
    rounds: int
    optimizer: str
    strategy: str
    batch_size: Optional[int] = 32
    steps_per_epoch: Optional[int] = 3

    class Config:
        arbitrary_types_allowed = True
