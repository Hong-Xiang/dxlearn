from dxl.fs import Path
from typing import Tuple
import numpy as np







class H5Data:
    @classmethod
    def split_file_data_path(full_path: Path) -> Tuple[Path, Path]:
        pass

    def __init__(self, file_path: Path, data_path: Path=None):
        file_path = Path(file_path)
        if data_path is None:
            file_path, data_path = self.split_file_data_path(file_path)
        self.file_path = file_path
        self.data_path = data_path

    def load(self, path: Path) -> np.ndarray:
        pass
