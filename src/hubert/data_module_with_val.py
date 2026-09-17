import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
import torchaudio
import torchaudio.transforms as T
import os
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import random

@dataclass(frozen=True)
class CSVColumns:
    audio_path: str = 'LINK'
    label: str = 'WORD'
    valid: str = 'VALID'

class AudioDataset(Dataset):
    """Dataset that loads audio files and processes waveforms on-the-fly."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        columns: CSVColumns = CSVColumns(),
        sample_rate=16000,
        max_length=None,
        audio_base_path=None,
        label_to_idx=None,
        transform=None,
    ):
        """
        Args:
            csv_path: Path to CSV file containing metadata
            columns: Relevant column names of the csv file
            sample_rate: Target sample rate for audio
            max_length: Maximum audio length in samples (None for no limit)
            audio_base_path: Base path to prepend to audio paths in CSV
            label_to_idx: Dictionary mapping labels to indices (auto-created if None)
            transform: Optional audio transform to apply (e.g., MFCC)
            k_shot: If specified, limits to k examples per class
            labels: If specified, only include these labels
        """
        self.df = df
        self.columns = columns
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.audio_base_path = audio_base_path
        self.transform = transform

        if label_to_idx is None:
            unique_labels = sorted(self.df[self.columns.label].astype(str).unique())
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx

        self.resampler = None
        
    
    def __len__(self):
        return len(self.df)
    
    def _load_audio(self, audio_path):
        """Load audio file and resample if necessary."""
        # Prepend base path if provided
        if self.audio_base_path:
            audio_path = os.path.join(self.audio_base_path, audio_path)
        
        # Load audio
        waveform, orig_sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if orig_sample_rate != self.sample_rate:
            if self.resampler is None or self.resampler.orig_freq != orig_sample_rate:
                self.resampler = T.Resample(
                    orig_freq=orig_sample_rate,
                    new_freq=self.sample_rate
                )
            waveform = self.resampler(waveform)
        
        # Trim or pad to max_length if specified
        if self.max_length is not None:
            if waveform.shape[1] > self.max_length:
                waveform = waveform[:, :self.max_length]
            elif waveform.shape[1] < self.max_length:
                padding = self.max_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        return waveform
    
    def __getitem__(self, idx):
        """Load audio, transform, and return with label."""
        # Get metadata
        row = self.df.iloc[idx]
        audio_path = row[self.columns.audio_path]
        label = row[self.columns.label]
        
        out = self._load_audio(audio_path)
        
        if self.transform is not None:
            out = self.transform(out)
        
        out = out.squeeze(0)  # (n_mfcc, time)
        
        label_idx = torch.tensor(self.label_to_idx[label])
        
        return out, label_idx


def collate_fn_audio(batch):
    """
    Collate function that pads MFCC sequences to the same length in a batch.
    Handles variable-length audio sequences.
    """
    mfccs, labels = zip(*batch)
    
    # Find max length in this batch
    max_len = max(mfcc.size(-1) for mfcc in mfccs)
    
    # Pad sequences
    padded_mfccs = []
    for mfcc in mfccs:
        if mfcc.size(-1) < max_len:
            pad_shape = list(mfcc.shape)
            pad_shape[-1] = max_len - mfcc.size(-1)
            padding = torch.zeros(*pad_shape)
            mfcc = torch.cat([mfcc, padding], dim=-1)
        padded_mfccs.append(mfcc)
    
    return torch.stack(padded_mfccs), torch.stack(labels)


class AudioDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for audio data loaded from CSV."""
    
    def __init__(
        self,
        train_csv=None,
        test_csv=None,
        train_files=None,
        test_files=None,
        columns: CSVColumns = CSVColumns(),
        sample_rate=16000,
        max_length=None,
        audio_base_path=None,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        use_variable_length=True,
        transform = None,
        k_shot_train=None,
        k_shot_test=None,
        k_shot_val=None,
        labels = None,
        max_classes=None,
    ):
        """
        Args:
            train_csv: Path to training CSV
            val_csv: Path to validation CSV
            test_csv: Path to test CSV
            columns: Column names for the input CSVs
            num_classes: Number of classes (auto-detected if None)
            n_mfcc: Number of MFCC coefficients
            sample_rate: Target sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
            max_length: Maximum audio length in samples (None for variable length)
            audio_base_path: Base path for audio files
            batch_size: Batch size
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory for faster GPU transfer
            use_variable_length: Whether to use variable-length collation
        """
        super().__init__()
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.train_files = train_files
        self.test_files = test_files
        self.columns = columns
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.audio_base_path = audio_base_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_variable_length = use_variable_length
        self.transform = transform
        self.k_shot_train = k_shot_train
        self.k_shot_test = k_shot_test
        self.k_shot_val = k_shot_val
        self.labels = labels
        self.max_classes = max_classes
        
        # Will be set during setup
        self.label_to_idx = None


    def _clean_and_filter_df_by_class(self, df: pd.DataFrame, k_shot = None) -> pd.DataFrame:

        invalid_rows = df[df[self.columns.valid].astype(bool).apply(lambda x: not x)].index
        df = df.drop(invalid_rows).reset_index(drop=True)

        if self.labels is not None:
            df = df[df[self.columns.label].isin(self.labels)].reset_index(drop=True)

        if k_shot is not None:
            df = df.groupby(self.columns.label).filter(lambda group: len(group) >= k_shot)

        return df

    def _retrieve_df_from_csv(self, csv_path, k_shot):
        if csv_path is None:
            return None
        
        df_full = pd.read_csv(csv_path)
        df = self._clean_and_filter_df_by_class(df_full, k_shot)
        return df
    
    def _get_unique_labels_from_df(self, df):
        if df is None:
            return None
        return set(df[self.columns.label].astype(str).unique())

    def _get_clean_dfs(self):

        assert not ((self.train_csv is None) and (self.test_csv is None)), "One cannot create datasets without at least one of train or test CSVs."
        clean_dfs = [None, None]

        csvs = [self.train_csv, self.test_csv]
        k_shots = [self.k_shot_train, self.k_shot_test]

        clean_dfs = [self._retrieve_df_from_csv(csv, k_shot) for csv, k_shot in zip(csvs, k_shots)]

        return clean_dfs

        
    def _make_label_mapping(self, clean_dfs):
        """Create label to index mapping from whichever CSV data we care about, taking filtering and shots into account."""

        labels_per_df = [self._get_unique_labels_from_df(df) for df in clean_dfs]
        unique_labels = set.intersection(*[labels for labels in labels_per_df if labels is not None])

        # Identify any specified labels that are missing
        if self.labels is not None and len(unique_labels) < len(self.labels):
            missing_labels = set(self.labels) - set(unique_labels)
            print(f"Warning: The following specified labels were not found in the datasets and will be ignored: {missing_labels}")

        # Optionally select a subset of the valid labels
        if self.max_classes is not None:
            unique_labels = set(sorted(list(unique_labels))[:self.max_classes])

        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        print(f"Created label to index mapping with {len(self.label_to_idx)} labels.")
        return
    

    def _sample_from_single_df(self, df: pd.DataFrame, k_shot=None) -> pd.DataFrame:
        """Select k shots per class from the dataframe."""
        if df is None:
            return None
        
        if k_shot is not None:
            df = df.groupby(self.columns.label).nth[:k_shot]
        return df
    
    def _filter_df_by_label_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataframe to only include labels in label_to_idx."""
        if df is None:
            return None
        filtered_df = df[df[self.columns.label].isin(self.label_to_idx.keys())].reset_index(drop=True)
        return filtered_df
    
    def _scrounge_validation_samples(self, used_files, filtered_clean_dfs):

        val_df = None
        if filtered_clean_dfs[0] is not None:
            print("Scrounging validation set from training data.")
            val_df = filtered_clean_dfs[0][~filtered_clean_dfs[0][self.columns.audio_path].isin(used_files)].reset_index(drop=True)
        elif filtered_clean_dfs[1] is not None:
            print("Scrounging validation set from test data.")
            val_df = filtered_clean_dfs[1][~filtered_clean_dfs[1][self.columns.audio_path].isin(used_files)].reset_index(drop=True)
        
        min_val_samples_per_class = val_df.groupby(self.columns.label).size().min()
        
        if (self.k_shot_val is None) or (min_val_samples_per_class < self.k_shot_val):
            print(f"Using {min_val_samples_per_class} validation samples per class.")
            self.k_shot_val = min_val_samples_per_class

        val_df = val_df.groupby(self.columns.label).nth[:self.k_shot_val]

        return val_df
    
    def _get_df_rows_by_path(self, df, paths):
        if df is None:
            return None
        filtered_df = df[df[self.columns.audio_path].isin(paths)].reset_index(drop=True)
        return filtered_df
    

    def _select_samples_for_datasets(self, clean_dfs) -> pd.DataFrame:
        """Get unused samples from train/test sets to make a validation set."""
        filtered_clean_dfs = [self._filter_df_by_label_mapping(df) for df in clean_dfs]

        # Either sample organically of get specific files
        if self.train_files is None:
            train_df = self._sample_from_single_df(filtered_clean_dfs[0], self.k_shot_train)
        else:
            train_df = self._get_df_rows_by_path(filtered_clean_dfs[0], set(self.train_files['link']))

        if self.test_files is None:
            test_df = self._sample_from_single_df(filtered_clean_dfs[1], self.k_shot_test)
        else:
            test_df = self._get_df_rows_by_path(filtered_clean_dfs[1], set(self.test_files['link']))

        test_k_values = test_df.groupby(self.columns.label).size().nunique()
        if test_k_values >1:
            test_df = self.standardise_test_shots(test_df)

        # Sort test df so that batching will work per-word
        test_df = test_df.sort_values(by=self.columns.label).reset_index(drop=True)

        used_files = set(test_df[self.columns.audio_path].tolist()).union(set(train_df[self.columns.audio_path].tolist()))

        val_df = self._scrounge_validation_samples(used_files,filtered_clean_dfs)

        return train_df, test_df, val_df
    
    def standardise_test_shots(self, df):

        grouped = df.groupby(self.columns.label)

        group_sizes = grouped.size()
        mode_size = group_sizes.mode()[0]
        
        # Process each group
        rows_to_add = []
        rows_to_remove = []
        
        for label, group in grouped:
            current_size = len(group)
            
            if current_size < mode_size:
                # Need to add rows - duplicate existing rows
                n_to_add = mode_size - current_size
                # Repeat rows cyclically to reach mode_size
                duplicates = group.iloc[:(n_to_add % current_size)]
                full_copies = [group] * (n_to_add // current_size)
                rows_to_add.extend(full_copies + [duplicates] if len(duplicates) > 0 else full_copies)
                
            elif current_size > mode_size:
                # Need to remove rows - keep only mode_size rows
                rows_to_keep = group.iloc[:mode_size]
                rows_to_remove.append(group.iloc[mode_size:].index)
        
        # Remove excess rows
        if rows_to_remove:
            indices_to_remove = pd.Index([]).union_many(rows_to_remove)
            df = df.drop(indices_to_remove)
        
        # Add duplicated rows
        if rows_to_add:
            df = pd.concat([df] + rows_to_add, ignore_index=False)
        
        # Sort by index to maintain some order
        df = df.sort_index()

        return df

    
    def setup(self, stage=None):
        """Set up datasets for each stage."""
        # First, create label mapping from train set
        clean_dfs = self._get_clean_dfs()
        self._make_label_mapping(clean_dfs)
        train_df, test_df, val_df = self._select_samples_for_datasets(clean_dfs)
        
        if stage == 'fit' or stage is None:
            if train_df is not None:
                self.train_dataset = AudioDataset(
                    df = train_df,
                    columns=self.columns,
                    sample_rate=self.sample_rate,
                    max_length=self.max_length,
                    audio_base_path=self.audio_base_path,
                    label_to_idx=self.label_to_idx,
                    transform=self.transform,
                )
            
            if val_df is not None:
                self.val_dataset = AudioDataset(
                    df = val_df,
                    columns=self.columns,
                    sample_rate=self.sample_rate,
                    max_length=self.max_length,
                    audio_base_path=self.audio_base_path,
                    label_to_idx=self.label_to_idx,
                    transform=self.transform,
                )

        if stage == 'test' or stage is None:
            if test_df is not None:
                self.test_dataset = AudioDataset(
                    df = test_df,
                    columns=self.columns,
                    sample_rate=self.sample_rate,
                    max_length=self.max_length,
                    audio_base_path=self.audio_base_path,
                    label_to_idx=self.label_to_idx,
                    transform=self.transform,
                )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=collate_fn_audio if self.use_variable_length else None
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, # TODO: less messy
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=collate_fn_audio if self.use_variable_length else None
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.k_shot_test,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=collate_fn_audio if self.use_variable_length else None,
        )