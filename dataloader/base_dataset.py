from abc import ABC, abstractmethod
from datasets import load_dataset


class BaseDataset(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.train_data, self.eval_data = self._load_data()
    
    @abstractmethod
    def get_dataset_name(self) -> str:
        pass

    @abstractmethod
    def _load_data(self):
        """
        Load the dataset from HuggingFace.
        Convert it to two columns question and answer for consistent manipulation.
        """
        pass

    def get_eval_data(self):
        return self.eval_data

    def get_train_data(self):
        return self.train_data

    def __iter__(self):
        self.__index = 0
        return self
    
    def __next__(self):
        if self.__index >= len(self.eval_data):
            raise StopIteration

        example = self.eval_data[self.__index]
        self.__index += 1
        return example

    def __len__(self):
        return len(self.eval_data) 


class GSM8K(BaseDataset):
    
    def get_dataset_name(self) -> str:
        """
        Should be same as the template yaml file. 
        Used to load the relevant template file fow now. 
        Can be later set in config.
        """
        return "GSM8K"

    def _load_data(self):
        train_data = load_dataset("openai/gsm8k", "main", split="train")
        eval_data = load_dataset("openai/gsm8k", "main", split="test")
        return train_data, eval_data
    
