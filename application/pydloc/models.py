from typing import Optional, List
from enum import Enum
from pydantic.main import BaseModel


class BasicConfiguration(BaseModel):
    config_id: Optional[str]
    batch_size: Optional[int] = 32
    steps_per_epoch: Optional[int] = 3
    epochs: int
    learning_rate: Optional[float] = 0.05


class StatusEnum(str, Enum):
    INACTIVE = 'INACTIVE'
    WAITING = 'WAITING'
    TRAINING = 'TRAINING'
    INTERRUPTED = 'INTERRUPTED'
    FINISHED = 'FINISHED'


class Status(BaseModel):
    status: StatusEnum = StatusEnum.INACTIVE
    round: int = 0

    class Config:
        use_enum_values = True


class TCTrainingConfiguration(BaseModel):
    # TODO: add more optional parameters here in order to be able to customize all strategies
    strategy: str
    model_id: str
    num_rounds: int
    min_fit_clients: int  # Minimum number of clients to be sampled for the next round
    min_available_clients: int
    timeout: int
    adapt_config: str
    blacklisted: int
    config: List[BasicConfiguration]

    class Config:
        arbitrary_types_allowed = True
