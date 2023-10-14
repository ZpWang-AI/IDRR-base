import numpy as np
import pandas as pd

from typing import *
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import Dataset


class RankingDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
    