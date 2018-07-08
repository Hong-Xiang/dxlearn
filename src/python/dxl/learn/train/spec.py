from typing import NamedTuple, List, Optional


    
class TrainSpec(NamedTuple):
    batch_size: int = 32
    optimizer: str = None
    learning_rate: float = 1e-3
    nb_units: List[int] = None
    nb_steps: int = 1000000
    load_step: Optional[int] = None
    path_save: str = './model'
    path_summary: str = './summary'


    @classmethod
    def from_dict(cls, dct):
        valid_inputs = {k: dct[k] for k in cls._fields if k in dct}
        return cls(**valid_inputs)

