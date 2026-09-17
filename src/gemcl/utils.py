
import os
import random
import pandas as pd
import torchaudio
import torchaudio.transforms as T
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
import torch
import itertools
from loguru import logger
import yaml
import json
import csv


def validate_classes(classes: list[str], config: dict, train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    """
    Ensures that each class (word) has at least the required number of samples in both the training and testing datasets.

    Args:
        classes (list[str]): List of class names (words) to validate.
        config (dict): Configuration dictionary containing 'train_shots' and 'test_shots'.
        train_df (pd.DataFrame): DataFrame containing training data information.
        test_df (pd.DataFrame): DataFrame containing testing data information.

    Returns:
        list[str]: List of valid classes that meet the sample/config requirements.
    """

    valid_classes = []
    train_word_rows = train_df['WORD'].value_counts()
    test_word_rows = test_df['WORD'].value_counts()

    for word in classes:
        
        # Count them
        train_count = train_word_rows.get(word, 0)
        test_count = test_word_rows.get(word, 0)

        # Only add to the new list if it passes BOTH checks
        if train_count >= config['train_shots'] and test_count >= config['test_shots']:
            valid_classes.append(word)

    return valid_classes

def meta_split(language : str, data_set_path: str, config: dict ,train_df : pd.DataFrame, test_df: pd.DataFrame, random_seed: int =42) -> tuple[list[str], list[str]]:
    """
    Assumes that all the words are the name of the folders in the clips directory of the language.
    This function splits the classes into training and testing sets for meta-learning.
    It returns two lists of class names: one for training and one for testing.
    Roughly 70% of classes are used for meta-training and 30% for meta-testing.

    Args:
        language (str): The language code (e.g., 'en' for English).
        data_set_path (str): The base path to the dataset.
        config (dict): Configuration dictionary containing 'tasks' key.
        train_df (pd.DataFrame): DataFrame containing training data information.
        test_df (pd.DataFrame): DataFrame containing testing data information.
        random_seed (int, optional): Seed for random number generator. Defaults to 42. 
    Returns:
        tuple[list[str], list[str]]: A tuple containing two lists:
            - train_classes: List of class names for meta-training.
            - test_classes: List of class names for meta-testing.
    """
    random.seed(random_seed)
    classes = [d for d in os.listdir(os.path.join(data_set_path, f"{language}/clips"))]
    valid_classes = validate_classes(classes, config, train_df, test_df)
    random.shuffle(valid_classes)
    split_point = int(0.7 * len(valid_classes)) # * Batsi: If anyone has a better idea for splitting, let me know.
    train_classes = valid_classes[:split_point]
    test_classes = valid_classes[split_point:]
    random.seed()  # Reset the random seed.
    return train_classes, test_classes, len(valid_classes), len(train_classes), len(test_classes)

def write_meta_splits(language: str, data_set_path: str, config: dict, split_csv_path: str = None, random_seed: int =42) -> None:
    """
    Write the meta-training and meta-testing class splits to text files.
    Args:
        language (str): The target language code (e.g., 'en').
        data_set_path (str): The root path to the dataset directory.
        split_csv_path (str, optional): Path to the CSV files. Defaults to None.
        config (dict): Configuration dictionary containing parameters for splitting.
        random_seed (int, optional): Seed for random number generator. Defaults to 42.
    Returns:
        None
    """

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

    meta_train_classes, meta_test_classes, valid_classes_count, train_classes_count, test_classes_count = meta_split(language, data_set_path, config, train_df, test_df)

    file_path = f"./meta_splits/train_{config['train_shots']}_shots_test_{config['test_shots']}_shots/{language}"
    os.makedirs(file_path, exist_ok=True)

    with open(os.path.join(file_path, "meta_train_classes.txt"), 'w') as f:
        f.write('\n'.join(meta_train_classes))

    with open(os.path.join(file_path, "meta_test_classes.txt"), 'w') as f:
        f.write('\n'.join(meta_test_classes))

    with open(os.path.join(file_path, "meta_data_classes.txt"), 'w') as f:
        f.write(f"The random seed used for splitting: {random_seed}\n")
        f.write(f"Total valid classes for language {language}: {valid_classes_count}\n")
        f.write(f"Meta-train classes for language {language}: {train_classes_count}\n")
        f.write(f"Meta-test classes for language {language}: {test_classes_count}")

def read_files(*filepaths: str) -> tuple[list[str]]:
    """
    Reads multiple text files and returns their contents as lists of strings.
    Args:
        *filepaths (str): Variable number of file paths to read.

    Returns:
        tuple[list[str]]: A tuple containing lists of strings from each file.
    """
    container = []
    for path in filepaths:
        try:
            with open(path, 'r') as f:
                data = [line.strip() for line in f if line.strip()]
                container.append(data)
        except FileNotFoundError:
            logger.warning(f"Warning: '{path}' not found.")
            container.append([])

    return tuple(container)

# * Example usage for generating splits for single language
# write_meta_splits("en", "./data/macroset", {"train_shots": 5, "test_shots": 5}, "./splits")


def prepare_mfccs(batch: list[list[str]]) -> torch.Tensor:

    resample_transform = T.Resample(orig_freq=48000, new_freq=16000) 
    episode_mfccs = []
    batch_of_episodes_mfccs = []
    transforms = T.MFCC(sample_rate=16000, n_mfcc=13, melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40, "center": False})

    for episode in batch:
        for path in episode:
            data_torch, og_sample_rate = torchaudio.load(path)
            resampled_audio = resample_transform(data_torch)  # Resample the audio to 16kHz. Resample audio has shape [1, 16k].
            transformed_audio = transforms(resampled_audio)
            transformed_audio = rearrange(transformed_audio, '1 n_mfcc t -> t (1 n_mfcc)') # for the encoder model. It expects this shape.
            episode_mfccs.append(transformed_audio)
        batch_of_episodes_mfccs.append(episode_mfccs)
        episode_mfccs = []

    batch_size = len(batch_of_episodes_mfccs)
    total_shots = len(batch_of_episodes_mfccs[0]) # the total shots is tasks*shots
    batch_of_episodes_mfccs = list(itertools.chain(*batch_of_episodes_mfccs))
    padded_flat = pad_sequence(batch_of_episodes_mfccs, batch_first=True, padding_value=0.0)

    batch_of_episodes_mfccs = rearrange(
        padded_flat, 
        '(batch total_shots) t f -> (batch total_shots) f t', 
        batch=batch_size, 
        total_shots=total_shots, 
        f=13,
        # t=98,
    )

    return batch_of_episodes_mfccs

def create_df_from_paths(paths: list[str]) -> pd.DataFrame:
    """
    Creates a DataFrame from a list of file paths, extracting the "word" from each path.
    Args:
        paths (list[str]): List of file paths.      
    Returns:
        pd.DataFrame: DataFrame with columns "link" and "word".
    """
    data = []
    for path in paths:
        # Logic: Extract the parent folder name as the "word"
        # * Example: "data/.../clips/batsi_is_awesome/file.opus" -> "batsi_is_awesome"
        parent_dir = os.path.dirname(path)
        word = os.path.basename(parent_dir)
        data.append({"link": path, "word": word})
    return pd.DataFrame(data)

def write_up_experiment_data( config: dict, x_train : list[list[str]], x_test : list[list[str]], split_type: str, single_language: bool = True, language: str = None) -> None:
    """
    Writes the training and testing data used in an experiment to text files for record-keeping.

    Args:
        config (dict): Configuration dictionary containing parameters for the experiment.
        x_train (list[list[str]]): Nested list containing training data file paths.
        x_test (list[list[str]]): Nested list containing testing data file paths.
        split_type (str): Type of split, e.g., "meta-train" or "meta-test".
        single_language (bool, optional): Indicates if the experiment is for a single language. Defaults to True.
        language (str, optional): Language code if single_language is True. Defaults to None.

    Returns:
        None
    """
    if single_language:
        assert language is not None, "Language must be specified when single_language is True."

    lang_prefix = language if single_language else "multi"
    folder_name = f"{lang_prefix}_{config['tasks']}-way-{config['train_shots']}-shot"
    base_path = os.path.join(f"./experiments/{split_type}", folder_name)
    
    os.makedirs(base_path, exist_ok=True)

    metadata_path = os.path.join(base_path, "meta-data.yaml")
    with open(metadata_path, 'w') as f:
        yaml.dump(config, f)

    # Iterate through each episode (batch index)
    num_episodes = len(x_train)
    
    for i in range(num_episodes):
        # Create a subfolder for this specific episode
        episode_dir = os.path.join(base_path, f"episode_{i}")
        os.makedirs(episode_dir, exist_ok=True)

        df_train = create_df_from_paths(x_train[i])
        df_train.to_csv(os.path.join(episode_dir, "train_links.csv"), index=False)
        
        df_test = create_df_from_paths(x_test[i])
        df_test.to_csv(os.path.join(episode_dir, "test_links.csv"), index=False)

def reading_json(json_file_path: str, output_csv_path: str = 'metadata.csv') -> None:
    """Reads metadata, counts unique recordings, and generates a sorted CSV.

    This function parses a language metadata file.

    Args:
        json_file_path (str): Path to the source JSON file.
        output_csv_path (str): Path for the output CSV. 
    Returns:
        None
    """
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []

    for key, meta_data in data.items():
        if not isinstance(meta_data, dict):
            continue

        # The 'filenames' dict looks like: { "word": ["file1.opus", "file2.opus"], ... }
        filenames = []
        
        filenames_dict = meta_data.get('filenames', {})
        
        if filenames_dict:
            # Loop through every list of files in the dictionary and add them to the list
            for file_list in filenames_dict.values():
                filenames.extend(file_list)
        
        recordings_count = len(filenames)

        row = {
            'language': meta_data.get('language', 'Unknown'),
            'language_key': key,
            'number_of_words': meta_data.get('number_of_words', 0),
            'number_of_recordings': recordings_count
        }
        rows.append(row)

    rows.sort(key=lambda x: x['number_of_recordings'])

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        headers = ['language', 'language_key', 'number_of_words', 'number_of_recordings']
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

def count_recordings(txt_path: str, csv_path: str, suffix: str) -> None:
    """
    Counts valid recordings for target words from a CSV file and writes summary and breakdown files.
    Args:
        txt_path (str): Path to the text file containing target words (one per line).
        csv_path (str): Path to the CSV file with recording metadata.
        suffix (str): Suffix for output filenames to distinguish different runs.
    Returns:
        None
    """
    with open(txt_path, 'r') as f:
        target_words = {line.strip() for line in f if line.strip()}
    
    # Initialise data structures
    # per_word_stats: {word: {'train': 0, 'dev': 0, 'test': 0, 'total': 0}}
    per_word_stats = {word: {'train': 0, 'dev': 0, 'test': 0, 'total': 0} for word in target_words}
    
    global_totals = {'train': 0, 'dev': 0, 'test': 0, 'total': 0}

    # 2. Process the CSV file
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row['WORD']
            # Normalize 'valid' and 'set' for case-insensitivity
            is_valid = row['VALID'].strip().lower() == 'true'
            current_set = row['SET'].strip().lower()
            
            if word in target_words and is_valid:
                # Update word-specific counts
                if current_set in per_word_stats[word]:
                    per_word_stats[word][current_set] += 1
                per_word_stats[word]['total'] += 1
                
                # Update global totals
                if current_set in global_totals:
                    global_totals[current_set] += 1
                global_totals['total'] += 1

    meta_filename = f"meta_count_{suffix}.txt"
    with open(meta_filename, 'w') as f:
        f.write(f"Total Valid Recordings: {global_totals['total']}\n")
        f.write(f"Train: {global_totals['train']}\n")
        f.write(f"Dev: {global_totals['dev']}\n")
        f.write(f"Test: {global_totals['test']}\n")

    breakdown_filename = f"word_breakdown_{suffix}.csv"
    with open(breakdown_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'total_valid', 'train', 'dev', 'test'])
        for word in sorted(per_word_stats.keys()):
            s = per_word_stats[word]
            writer.writerow([word, s['total'], s['train'], s['dev'], s['test']])
# Example usage:
# count_recordings('./meta_splits/train_5_shots_test_5_shots/en/meta_train_classes.txt', './splits/en/en_splits.csv', 'meta_train')
# count_recordings('./meta_splits/train_5_shots_test_5_shots/en/meta_test_classes.txt', './splits/en/en_splits.csv', 'meta_test')

def process_episode_times(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Processes episode times from a CSV file and writes the summary to a new CSV file.

    Args:
        input_path (str): Path to the input CSV file containing episode times.
        output_path (str): Path to the output CSV file to save the summary.

    Returns:
        pd.DataFrame: DataFrame containing the summary of episode times.
    """
    df = pd.read_csv(input_path)
    
    # Calculate the total time per episode
    # Grouping by 'Episode' and summing the 'Total_Lesson_Block_Seconds' column
    summary = df.groupby('Episode')['Total_Lesson_Block_Seconds'].sum().reset_index()
    
    # Write the result to a new CSV file
    summary.to_csv(output_path, index=False)
    return summary

# Example usage:
# process_episode_times('./experiments/meta-test/en_1000-way-5-shot/learn_class_stats_times.csv', 'gemcl_time_to_learn_classes_mfcc.csv')

def generate_inference_reports(file_path: str) -> None:
    """
    Generates inference reports from a CSV file containing test times.

    Args:
        file_path (str): Path to the CSV file containing test times.

    Returns:
        None
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # We group by both Episode and Classes_Learned to sum up all task_ids
    ep_summary = df.groupby(['episode', 'classes_learned'])['test_time'].sum().reset_index()
    ep_summary.rename(columns={'test_time': 'total_test_time'}, inplace=True)
    
    # Save the episode-specific file
    ep_filename = "gemcl_total_test_per_episode.csv"
    ep_summary.to_csv(ep_filename, index=False)
    
    # We take the results from Step 1 and average them by the class count
    global_avg = ep_summary.groupby('classes_learned')['total_test_time'].mean().reset_index()
    global_avg.rename(columns={'total_test_time': 'mean_total_test_time'}, inplace=True)
        
    global_filename = "gemcl_avg_time_across_episodes.csv"
    global_avg.to_csv(global_filename, index=False)

# generate_inference_reports('./experiments/meta-test/en_1000-way-5-shot/task_experimental_results.csv')