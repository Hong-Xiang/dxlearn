from doufo import dataclass
from typing import Dict, Sequence


@dataclass
class JobSpec:
    ip: str
    port: int


@dataclass
class ClusterSpec:
    jobs: Dict[str, Sequence[JobSpec]]
