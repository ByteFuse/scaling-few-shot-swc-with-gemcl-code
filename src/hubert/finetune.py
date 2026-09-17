import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from data_module_with_val import AudioDataModule, CSVColumns
import yaml
import os
from box import Box
import argparse
import pandas as pd
import torch

from processing_utils import read_formatted_csv, get_first_x_words

from hubert_pretrain_model import HubertClassifierLightningModule

def create_parser():
    """
    Create argument parser for training configuration.
    """
    parser = argparse.ArgumentParser(
        description='Training configuration parser',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # General training parameters
    parser.add_argument('--seed_everything', type=int, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for optimizer')
    parser.add_argument('--max_epochs', type=int, help='Maximum number of training epochs')
    parser.add_argument('--num_workers', type=int, help='Number of workers for data loading')
    parser.add_argument('--freeze_feature_encoder', type=lambda x: str(x).lower() == 'true', help='Whether to freeze the feature encoder')
    parser.add_argument('--freeze_base_model', type=lambda x: str(x).lower() == 'true', help='Whether to freeze the feature encoder')
    parser.add_argument('--max_classes', type=int, help='Maximum number of classes')
    parser.add_argument('--patience', type=int, help='Steps of nonimprovement before axeing exp.')
    
    # Audio processing parameters
    parser.add_argument('--sample_rate', type=int, help='Audio sample rate in Hz')
    parser.add_argument('--max_audio_length', type=int, help='Maximum audio length in samples (None for no limit)')
    parser.add_argument('--k_shot_test', type=int, help='Number of shots for testing')
    parser.add_argument('--k_shot_train', type=int, help='Number of shots for training')
    parser.add_argument('--k_shot_val', type=int, help='Number of shots for validation')
    parser.add_argument('--use_specific_words', type=lambda x: str(x).lower() == 'true', help='Whether to use specific words')
    
    # File paths
    parser.add_argument('--specific_words_relpath', type=str, help='Relative path to specific words file')
    parser.add_argument('--train_csv_relpath', type=str, help='Relative path to training CSV file')
    parser.add_argument('--labels_to_use_path', type=str, help='Relative path to labels CSV file')
    parser.add_argument('--val_csv_relpath', type=str, help='Relative path to validation CSV file')
    parser.add_argument('--audio_base_relpath', type=str, help='Relative path to audio clips directory')
    parser.add_argument('--experiment_name', type=str, help='Name for identification')
    parser.add_argument('--experiment_stage', type=str, help='Tuning/final/testing')
    

    # Most NB
    parser.add_argument('--config_path', type=str, default='classification/finetune_config.yaml', help='Relative path to config')

    
    return parser

paths = {
    42: 'src/gemcl/experiments/meta-test/en_1000-way-5-shot/episode_0/',
    3: 'src/gemcl/experiments/meta-test/en_1000-way-5-shot/episode_1/',
    1075: 'src/gemcl/experiments/meta-test/en_1000-way-5-shot/episode_2/',
    495: 'src/gemcl/experiments/meta-test/en_1000-way-5-shot/episode_3/',
    248: 'src/gemcl/experiments/meta-test/en_1000-way-5-shot/episode_4/',
    486: 'src/gemcl/experiments/meta-test/en_1000-way-5-shot/episode_5/',
    3840: 'src/gemcl/experiments/meta-test/en_1000-way-5-shot/episode_6/',
    5037: 'src/gemcl/experiments/meta-test/en_1000-way-5-shot/episode_7/',
    5: 'src/gemcl/experiments/meta-test/en_1000-way-5-shot/episode_8/',
    59: 'src/gemcl/experiments/meta-test/en_1000-way-5-shot/episode_9/',
}

def main():
    parser = create_parser()
    args = parser.parse_args()

    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.CLoader) # TODO get this somehow else
    
    config.update({k: v for k, v in vars(args).items() if v is not None})
    config = Box(config)

    # Set random seed for reproducibility
    pl.seed_everything(config.seed_everything)

    # Update paths
    here = os.path.dirname(os.path.abspath(__file__))
    path_prefix = os.path.join(here, '../../')
    train_csv_path = os.path.join(path_prefix, config.train_csv_relpath)
    val_csv_path = os.path.join(path_prefix, config.val_csv_relpath)
    audio_base_path = os.path.join(path_prefix, config.audio_base_relpath)
    # full_labels_to_use_path = os.path.join(path_prefix, config.labels_to_use_path)

    experiment_id = f"{config.experiment_name}_{config.experiment_stage}_{config.learning_rate}_{config.max_classes}_{config.seed_everything}"
    
    # ======================================================
    
    # Create data module
    print("\nInitializing data module...")

    columns = CSVColumns(
        audio_path='LINK',
        label='WORD',
        valid='VALID'
    )

    specific_files_base_path = paths[config.seed_everything]

    train_files_to_use = read_formatted_csv(os.path.join(path_prefix,specific_files_base_path,'train_links.csv'))
    test_files_to_use = read_formatted_csv(os.path.join(path_prefix,specific_files_base_path,'test_links.csv'))

    words_to_use = get_first_x_words(train_files_to_use, config.max_classes)

    data_module = AudioDataModule(
        train_csv=train_csv_path,
        test_csv=val_csv_path,
        test_files=test_files_to_use,
        train_files=train_files_to_use,
        columns=columns,
        sample_rate=config.sample_rate,
        max_length=config.max_audio_length,
        audio_base_path=audio_base_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        use_variable_length=True,  # Handle variable-length sequences
        transform=None, 
        k_shot_train = config.k_shot_train,
        k_shot_test=config.k_shot_test,
        k_shot_val=config.k_shot_val,
        labels=words_to_use,
        max_classes= config.max_classes,
    )
    
    # Setup to get dataset info
    data_module.setup("fit")

    num_classes = len(data_module.label_to_idx)
    
    # Calculate max steps for scheduler
    steps_per_epoch = len(data_module.train_dataset) // config.batch_size
    max_steps = steps_per_epoch * config.max_epochs
    
    print(f"\nDataset info:")
    print(f"  Training samples: {len(data_module.train_dataset)}")
    print(f"  Validation samples: {len(data_module.val_dataset)}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {max_steps}")
    
    # Create model
    model = HubertClassifierLightningModule(
        model_name="facebook/hubert-base-ls960",
        num_labels=num_classes,  # e.g., 10 audio classes
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        max_epochs=config.max_epochs,
        freeze_feature_encoder=config.freeze_feature_encoder,
        freeze_layers=0,
        freeze_base_model=config.freeze_base_model,
    )
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'classification/experiment_{config.experiment_name}/{config.experiment_stage}/{experiment_id}',
        filename='hubert-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=False,
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        patience=config.patience,
        mode='min',
        verbose=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator='auto',  # Automatically uses GPU if available
        devices=1,
        precision='16-mixed',  # Mixed precision training
        callbacks=[early_stop_callback, lr_monitor, checkpoint_callback],
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        log_every_n_steps=10,
        deterministic=False,  # Set to True for reproducibility (slower)
        enable_checkpointing=True,
    )
    
    trainer.fit(model, data_module)
    print("\nTraining complete!")

    # Run best model on the test set
    print("\nEvaluating best model on test set...")
    test_dir = f'classification/experiment_{config.experiment_name}/{config.experiment_stage}/'
    version = 'best_model'

    manual_save_path = os.path.join(test_dir, experiment_id, version)

    csv_logger = CSVLogger(test_dir, name=experiment_id, version=version)
    csv_logger.log_hyperparams(params = config)
    label_to_idx_df = pd.DataFrame(data_module.label_to_idx.items(), columns=['WORD', 'INDEX'])

    best_model = HubertClassifierLightningModule(
        model_name="facebook/hubert-base-ls960",
        num_labels=num_classes,  # e.g., 10 audio classes
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        max_epochs=config.max_epochs,
        freeze_feature_encoder=config.freeze_feature_encoder,
        freeze_layers=0,
    )
    best_state_dict = torch.load(checkpoint_callback.best_model_path)
    best_model.load_state_dict(best_state_dict['state_dict'])

    # best_model = HubertClassifierLightningModule.load_from_checkpoint(checkpoint_callback.best_model_path, weights_only=False)

    tester = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator='auto',  # Automatically uses GPU if available
        devices=1,
        precision='16-mixed',  # Mixed precision training
        callbacks=None,
        logger=csv_logger,
        log_every_n_steps=10,
        deterministic=True,
        enable_checkpointing=False,
    )
    tester.test(best_model, datamodule=data_module)

    # a little bit of postprocessing to get the test results in a nice format
    results_df = pd.read_csv(os.path.join(manual_save_path,'metrics.csv'),)
    idx_to_label = {value: key for key, value in data_module.label_to_idx.items()}
    results_df[columns.label] = results_df['test_label'].map(idx_to_label)
    
    results_df.to_csv(os.path.join(manual_save_path,'final_test_results.csv'), index=False)
    label_to_idx_df.to_csv(os.path.join(manual_save_path,'word_to_index.csv'), index=False)
    print("Test results saved.")

    return model, trainer


if __name__ == "__main__":
    model, trainer = main()