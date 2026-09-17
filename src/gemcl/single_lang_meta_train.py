from argparse import ArgumentParser
import os
from torch.utils.data import DataLoader
from meta_dataset import MetaCommonWordsSingleLanguage as MetaDataset
from model.gemcl import GeMCL
import torch
from glob import glob
from os import path
from utils import prepare_mfccs
from loguru import logger

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

config ={
    'tasks': 25,
    'train_shots':5,
    'test_shots':5,
    'batch_size':16,
    'max_train_steps':5_000,
    'checkpoint_interval':2_500,
    'log_dir':'./checkpoints/meta_train_wav2vec2/25way_5shot',
    'lr': 5e-5,
    'weight_decay':1e-2,
    'bite':1,  # gradient accumulation bite size
}


parser = ArgumentParser()
parser.add_argument('--model_filename', type=str, default='meta_train.pt')
parser.add_argument('--language', type=str, default='en')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--resume', action='store_true')
args = parser.parse_args()

config['seed'] = args.seed
config['model_filename'] = args.model_filename

def meta_train(config: dict) -> None:
    """
    Args:
        config (dict): Configuration dictionary containing training parameters.
    Returns:
        None

    Meta-trains a GeMCL model using a specified configuration. Trains the model on N-way K-shot tasks (not in a continual-learning way).
    Logs training progress to terminal and saves model checkpoints at specified intervals.
    Makes use of gradient accumulation for large batch sizes.
    """
    torch.manual_seed(args.seed) # Setting the seed
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # GPU operations have a separate seed we also want to set
    # * Batsi: I am just a big fan of reproducibility.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Additionally, some operations on a GPU are implemented stochastic for efficiency.
    # We want to ensure that all operations are deterministic on GPU for reproducibility.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Initialize the Model
    model = GeMCL(config)
    model = model.to(device)
    optim = torch.optim.AdamW(lr=config['lr'], params=model.parameters(), weight_decay=config['weight_decay'])
    model.train()

    # Resume from Checkpoint if specified
    if args.resume:
        checkpoint_files = sorted(glob(path.join(config['log_dir'], 'step_*.pt')))
        latest_checkpoint = checkpoint_files[-1]
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        optim.zero_grad()
        logger.info(f"Resumed from checkpoint: {latest_checkpoint} at step {start_step + 1}")

    else:
        start_step = 0 

    # Load the Data
    meta_train_path = f"../../meta_splits/train_{config['train_shots']}_shots_test_{config['test_shots']}_shots/{args.language}/meta_train_classes.txt"
    meta_test_path = f"../../meta_splits/train_{config['train_shots']}_shots_test_{config['test_shots']}_shots/{args.language}/meta_test_classes.txt"
    meta_train_set = MetaDataset(args.language, '../../data/macroset', 'train', config, meta_train_path, meta_test_path, '../../splits')
    meta_train_loader = DataLoader(
        meta_train_set,
        batch_size=config['batch_size'],
        collate_fn=meta_train_set.episode_collate_fn)
    meta_train_loader_iter = iter(meta_train_loader)

    # Main Training Loop
    os.makedirs(config['log_dir'], exist_ok=True)
    
    for step in range(start_step + 1, config['max_train_steps'] + 1):
        train_x, train_y, test_x, test_y = next(meta_train_loader_iter)

        batch_size = config['batch_size']
        digested = 0

        losses = []
        accs = []

        # gradient accumulation
        while batch_size - digested > 0:
            train_x_bite = train_x[digested:digested + config['bite']]
            train_y_bite = train_y[digested:digested + config['bite']]
            test_x_bite = test_x[digested:digested + config['bite']]
            test_y_bite = test_y[digested:digested + config['bite']]
            
            train_x_bite = prepare_mfccs(train_x_bite)
            test_x_bite = prepare_mfccs(test_x_bite)

            train_x_bite = train_x_bite.to(device)
            test_x_bite = test_x_bite.to(device)
            train_y_bite = train_y_bite.to(device)
            test_y_bite = test_y_bite.to(device)

            loss, (acc, _) = model(train_x_bite, train_y_bite, test_x_bite, test_y_bite)

            # This is for logging purposes
            losses.append(loss.sum().detach())
            accs.append(acc.sum().detach())

            loss = loss.sum() / batch_size  # Normalise the loss to account for gradient accumulation
            loss.backward()
            digested += config['bite']

        optim.step()
        optim.zero_grad()
        avg_loss = torch.stack(losses).sum() / batch_size
        avg_acc = torch.stack(accs).sum() / batch_size

        logger.info(f"step={step} | loss={avg_loss:.4f} | acc={avg_acc:.4f}")



        if step % config['checkpoint_interval'] == 0:

            old_ckpt_paths = sorted(glob(path.join(config['log_dir'], 'step_*.pt')))

            # Save the model checkpoint
            checkpoint_path = f"{config['log_dir']}/step_{step}.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'config': config
            }, checkpoint_path)
            
            logger.info(f"Saved checkpoint at step {step} to {checkpoint_path}")

            # Remove old checkpoints
            for ckpt_path in old_ckpt_paths:
                os.remove(ckpt_path)

    logger.success("Meta-Training Complete.")

meta_train(config)
