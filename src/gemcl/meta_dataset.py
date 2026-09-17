import os
import random
from loguru import logger
from torch.utils.data import IterableDataset
import pandas as pd
import itertools
import torch
from einops import repeat
from utils import read_files

def get_y_label(tasks: int, shots: int) -> torch.Tensor:
    """
    Generates labels for the support and query sets in a few-shot learning task.
    Args:
        tasks (int): Number of classes (tasks).
        shots (int): Number of samples per class (shots).
    Returns:
        torch.Tensor: A tensor of shape (tasks * shots,) containing labels from 0 to tasks-1.
    """
    return repeat(torch.arange(tasks), 't -> (t s)', s=shots)


def get_valid_links(links_dict: dict[str, list[str]], shots: int, classes: list[str]) -> list[str]:
    """
    Returns the valid set of links for each word in the dataset.
    Each word must have at least 'shots' number of valid links.
    
    Args:
        links_dict (dict[str, list[str]]): A dictionary mapping words to their list of valid links.
        shots (int): The number of shots per class.
        classes (list[str]): The list of classes (words) to check.

    Returns:
        list[str]: A flat list of valid links for all classes.
    """
    
    list_of_links = []

    for word in classes:
        all_links = links_dict.get(word)
        
        # Sample
        selected_links = random.sample(all_links, shots) 
        list_of_links.append(selected_links)

    # just flattening the list    
    list_of_links = list(itertools.chain(*list_of_links)) # ? is this the best way to do it?
    return list_of_links

def get_samples_for_class_single_language(dataset: IterableDataset, classes: list[str]) -> list[list[str]]:
    """
    Each word from any lanuages has at least 1 example (duh).
    This function returns all the samples for a given class (word) in the specified language.
    Returns the samples for the support set and the query set of the class.

    Args:
        dataset (MetaCommonWordsSingleLanguage): The dataset object containing the dataframes and config.
        classes (list[str]): The list of classes (words) to get samples for.
    Returns:
        list[list[str]]: A tuple containing four elements:
            - train_links: List of training links for the support set.
            - train_y: Tensor of training labels for the support set.
            - test_links: List of testing links for the query set.
            - test_y: Tensor of testing labels for the query set
    """

    valid_links_train = get_valid_links(dataset.train_links_dict, dataset.config['train_shots'], classes)
    valid_links_test = get_valid_links(dataset.test_links_dict, dataset.config['test_shots'], classes)

    train_y = get_y_label(dataset.config['tasks'], dataset.config['train_shots'])
    test_y = get_y_label(dataset.config['tasks'], dataset.config['test_shots'])

    return valid_links_train, train_y, valid_links_test, test_y

class MetaCommonWordsSingleLanguage(IterableDataset):
    """
    An IterableDataset for Few-Shot Learning that streams episodes of audio clips.

    This dataset loads validation and testing CSVs for a specific language, filters for valid
    clips, and splits classes into meta-training and meta-testing sets. It then streams
    N-way K-shot episodes (tasks) based on the provided configuration.

    Args:
        language (str): The target language code (e.g., 'en').
        data_set_path (str): The root path to the dataset directory.
        split_type (str): Determines which set of classes to sample from:
            - 'train': Samples from the meta-training class pool.
            - 'test': Samples from the meta-testing class pool.
        config (dict): A dictionary containing episode configuration:
            - 'tasks' (int): The N-way (number of classes) per episode.
            - 'train_shots' (int): The K-shot (number of support samples) per class.
            - 'test_shots' (int): The number of query samples per class.
        meta_train_path (str): Path to the file containing meta-training class names.
        meta_test_path (str): Path to the file containing meta-testing class names.
        split_csv_path (str, optional): Path to the directory containing train/test CSV files. Defaults to None.
    """
    def __init__(self, language: str, data_set_path: str, split_type: str, config: dict, meta_train_path: str, meta_test_path: str, split_csv_path: str = None):
        super().__init__()
        # Getting the valid recordings from the csv files.
        split_csv_path = data_set_path if split_csv_path is None else split_csv_path
        test_path = os.path.join(split_csv_path, f"{language}/{language}_test.csv")
        train_path = os.path.join(split_csv_path, f"{language}/{language}_train.csv")

        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        df_train['VALID_CLEAN'] = df_train['VALID'].astype(str).str.lower() == 'true'
        df_test['VALID_CLEAN'] = df_test['VALID'].astype(str).str.lower() == 'true'

        # Filter global data first (get only rows where VALID is True)
        train_df = df_train[df_train['VALID_CLEAN'] == True]
        test_df = df_test[df_test['VALID_CLEAN'] == True]


        # making dicts for fast lookups.
        self.train_links_dict = train_df.groupby('WORD')['LINK'].apply(list).to_dict()
        self.test_links_dict = test_df.groupby('WORD')['LINK'].apply(list).to_dict()

        del df_train, df_test, train_df, test_df
        
        meta_train_classes, meta_test_classes = read_files( meta_train_path, meta_test_path)

        if split_type == 'train':
            check_failed = len(meta_train_classes) < config['tasks'] or len(meta_test_classes) < config['tasks']
        else:
            check_failed = len(meta_test_classes) < config['tasks']

        if check_failed:
                logger.error(
                    "STOPPING: Not enough classes after cleaning data.\n"
                    "Action: Reduce tasks OR change split ratio.\n"
                    f"Requested Tasks: {config['tasks']}\n"
                    f"Current Train Classes: {len(meta_train_classes)}\n"
                    f"Current Test Classes:  {len(meta_test_classes)}"
                )
                raise ValueError("Not enough classes to sample tasks.")
        
        self.config = config
        self.language = language
        self.data_set_path = data_set_path
        self.collate_fn = None

        if split_type == 'train':
            self.classes = meta_train_classes
        else:
            self.classes = meta_test_classes


    def __iter__(self):
        return self
    
    def __next__(self):
        assert self.config['tasks'] <= len(self.classes), "The number of sampled classes should be less than or equal to the total number of classes."
        classes = random.sample(self.classes, self.config['tasks'])
        train_links, train_y, test_links, test_y = get_samples_for_class_single_language(self, classes)

        train_list, test_list=[],[]

        # Iterate through your list of lists
        for train_link in train_links:
            # Construct the path
            train_full_path = os.path.join(self.data_set_path, f"{self.language}/clips", train_link)
            train_list.append(train_full_path)

        for test_link in test_links:
            # Construct the path for the test link
            test_full_path = os.path.join(self.data_set_path, f"{self.language}/clips", test_link)
            test_list.append(test_full_path)
                
        return train_list, train_y, test_list, test_y

    def episode_collate_fn(self, batch: list) -> tuple[list[list[str]], torch.Tensor, list[list[str]], torch.Tensor]:
        """
        Custom collate function to keep episodes grouped correctly.
        
        Args:
            batch (list): A list of tuples returned by __next__.
                        Structure: [(train_x, train_y, test_x, test_y), ...]
        
        Returns:
            tuple: Grouped batches where each element is a list/tensor of episodes.
                x output shape: [batch_size, n_shots_total] (List of strings)
                y output shape: [batch_size, n_shots_total] (Tensor)
        """
        # batch is a list of N samples (where N = batch_size)
        # Each sample is (train_x_list, train_y_tensor, test_x_list, test_y_tensor)
        
        # We simply collect the lists. 
        train_x_batch = [item[0] for item in batch]
        
        train_y_batch = torch.stack([item[1] for item in batch])
        
        test_x_batch = [item[2] for item in batch]
        
        test_y_batch = torch.stack([item[3] for item in batch])
        return train_x_batch, train_y_batch, test_x_batch, test_y_batch
    