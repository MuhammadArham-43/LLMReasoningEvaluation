from abc import ABC, abstractmethod
from datasets import load_dataset, Dataset
import pandas as pd


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
    

# ------ 5- few shot samples from set of 35 examples (good representative + <10 % of dataset leakage) -----

class MATH500(BaseDataset):

    def get_dataset_name(self) -> str:
        """
        Should be same as the template yaml file. 
        Used to load the relevant template file for now. 
        Can be later set in config.
        """
        return "MATH500"

    # def _load_data(self):
    #     train_data = None  # No train split available
    #     eval_data = load_dataset("HuggingFaceH4/MATH-500", split="test")
    #     return train_data, eval_data
        
    def _load_data(self):
        dataset = load_dataset("HuggingFaceH4/MATH-500")
        df = dataset['test'].to_pandas()
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        selected_rows = []
        used_indices = set()

        subjects = ['Geometry', 'Prealgebra', 'Counting & Probability', 
                    'Algebra', 'Number Theory', 'Precalculus', 'Intermediate Algebra']
        levels = [1, 2, 3, 4, 5]

        # Select 1 example per subject per level (total 35 for 5 few-shot)
        for subject in subjects:
            for level in levels:
                subset = df[(df['subject'] == subject) & (df['level'] == level)]
                if len(subset) == 0:
                    raise ValueError(f"No example found for subject {subject} and level {level}")
                row = subset.sample(n=1, random_state=42)  # random but reproducible
                selected_rows.append(row)
                used_indices.update(row.index)

        train_df = pd.concat(selected_rows)
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Remaining examples are eval set
        eval_df = df[~df.index.isin(used_indices)].reset_index(drop=True)

        train_data = Dataset.from_pandas(train_df)
        eval_data = Dataset.from_pandas(eval_df)

        return train_data, eval_data


# ------ 5- few shot samples from set of 35 examples (good representative + <10 % of dataset leakage) -----
class MathOdyssey(BaseDataset):

    def get_dataset_name(self) -> str:
        """
        Should be same as the template yaml file. 
        Used to load the relevant template file for now. 
        Can be later set in config.
        """
        return "MathOdyssey"

    def _load_data(self):
        # Load dataset
        dataset = load_dataset("YourOrg/MathOdyssey")
        df = dataset['test'].to_pandas()
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Select 30 random samples for train (15 categories so chose 30)
        train_df = df.sample(n=30, random_state=42).reset_index(drop=True)

        # Remaining for eval/test
        eval_df = df.drop(train_df.index).reset_index(drop=True)

        train_data = Dataset.from_pandas(train_df)
        eval_data = Dataset.from_pandas(eval_df)

        return train_data, eval_data

# ------ 3-few shot samples from set of 5 examples (good representative + ~15-16 % of dataset leakage) -----
class AIME2024(BaseDataset):

    def get_dataset_name(self) -> str:
        """
        Should be same as the template yaml file. 
        Used to load the relevant template file for now. 
        Can be later set in config.
        """
        return "AIME2024"

    def _load_data(self):
        # Load dataset
        dataset = load_dataset("Maxwell-Jia/AIME_2024")
        df = dataset['test'].to_pandas()

        # Shuffle dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Select 5 samples for train set
        train_df = df.iloc[:5].reset_index(drop=True)

        # Remaining 25 samples for eval/test set
        eval_df = df.iloc[5:].reset_index(drop=True)

        train_data = Dataset.from_pandas(train_df)
        eval_data = Dataset.from_pandas(eval_df)

        return train_data, eval_data


# ------ 3-few shot samples from set of 5 examples (good representative + ~15-16 % of dataset leakage) -----
class AIME2025(BaseDataset):

    def get_dataset_name(self) -> str:
        """
        Should be same as the template yaml file. 
        Used to load the relevant template file for now. 
        Can be later set in config.
        """
        return "AIME2025"

    def _load_data(self):
        # Load dataset
        dataset = load_dataset("math-ai/aime25")
        df = dataset['test'].to_pandas()

        # Shuffle dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Select 5 samples for train set
        train_df = df.iloc[:5].reset_index(drop=True)

        # Remaining 25 samples for eval/test set
        eval_df = df.iloc[5:].reset_index(drop=True)

        train_data = Dataset.from_pandas(train_df)
        eval_data = Dataset.from_pandas(eval_df)

        return train_data, eval_data