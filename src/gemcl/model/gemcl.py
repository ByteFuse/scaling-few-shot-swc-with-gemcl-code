# Based on the implementation from https://github.com/soochan-lee/SB-MCL/blob/main/models/gemcl.py

from typing import Tuple
import torch
import torch.nn as nn
from einops import rearrange, reduce
from model.encoder import Wav2Vec2Model


class GeMCL(nn.Module):
    """Prototypical Network"""

    def __init__(self, config: dict, continual_learning: bool = False):
        super().__init__()
        self.x_encoder = Wav2Vec2Model(use_feature_encoder=False)
        self.ce = nn.CrossEntropyLoss(reduction='none')

        self.alpha = nn.Parameter(torch.tensor(100.), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(1000.), requires_grad=True)
        self.config = config
        self.continual_learning = continual_learning

        if self.continual_learning:
            self.prototypes = None
            self.var = None
            self.squared_diff = None

    def encode_x(self, train_x: torch.Tensor, test_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        train_num = train_x.shape[0]
        # batch, test_num = test_x.shape[:2]

        x = torch.cat([train_x, test_x], dim=0) # * I am assuming processing all the data at once is more efficient.
        x_enc = self.x_encoder(x) # ? Just noticed that the layer norm statistics are calculated using the test and train data together. Is this correct? Think so.
        # train_x_enc, test_x_enc = torch.split(x_enc, [train_num, test_num], dim=1)
        train_x_enc = x_enc[:train_num]
        test_x_enc = x_enc[train_num:]
        return train_x_enc, test_x_enc

    def forward(self, train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, test_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        batch, test_num = test_y.shape
        train_x_enc, test_x_enc = self.encode_x(train_x, test_x) # shape(batch tasks shots) seq_len hidden_dim for both
        train_x_enc = rearrange(train_x_enc, '(batch tasks shots) seq_len hidden_dim -> batch (tasks shots) (seq_len hidden_dim)', batch=self.config['bite'], tasks=self.config['tasks'], shots=self.config['train_shots'])
        test_x_enc = rearrange(test_x_enc, '(batch tasks shots) seq_len hidden_dim -> batch (tasks shots) (seq_len hidden_dim)', batch=self.config['bite'], tasks=self.config['tasks'], shots=self.config['test_shots'])
        # train_x_enc, test_x_enc = train_x, test_x

        # Train
        prototypes = reduce(
            train_x_enc, 'b (t s) d -> b t d', 'mean', t=self.config['tasks'], s=self.config['train_shots'])
        squared_diff = (
                rearrange(train_x_enc, 'b (t s) d -> b t s d', t=self.config['tasks']) -
                rearrange(prototypes, 'b t d -> b t 1 d')
        ).square()
        squared_diff = reduce(squared_diff, 'b t s d -> b t d', 'sum')

        alpha_prime = self.alpha + self.config['train_shots'] / 2
        beta_prime = self.beta + squared_diff / 2

        var = beta_prime / alpha_prime * (1 + 1 / self.config['train_shots'])

        # Test
        test_x_enc = rearrange(test_x_enc, 'b l h -> b l 1 h')
        prototypes = rearrange(prototypes, 'b t h -> b 1 t h')
        squared_diff = (test_x_enc - prototypes).square()  # b l t h
        var = rearrange(var, 'b t h -> b 1 t h')

        eps = 1e-8
        nll = (squared_diff / (var * alpha_prime * 2) + 1).log() * (alpha_prime + 0.5) + (var + eps).log()
        nll = reduce(nll, 'b l t h -> b l t', 'sum')

        logit = -nll
        loss = self.ce(rearrange(logit, 'b l t -> (b l) t'), rearrange(test_y, 'b l -> (b l)'))
        loss = reduce(loss, '(b n) -> b', 'mean', b=batch, n=test_num)

        return loss, standard_classification_calculation(logit, test_y, self.config['test_shots'])
    
    def learn_class_statistics(self, train_x: torch.Tensor) -> None:

        train_x_enc = self.x_encoder(train_x) # shape(batch tasks shots) seq_len hidden_dim
        train_x_enc = rearrange(train_x_enc, '(batch tasks shots) seq_len hidden_dim -> batch (tasks shots) (seq_len hidden_dim)', batch=self.config['episode_bite'], tasks=self.config['sub_tasks'], shots=self.config['train_shots'])
        # train_x_enc, test_x_enc = train_x, test_x

        # The mean of the samples for the current sub-tasks
        sub_prototypes = reduce(
            train_x_enc, 'b (t s) d -> b t d', 'mean', t=self.config['sub_tasks'], s=self.config['train_shots'])
        
        # Update the prototypes by concatenating the new sub_prototypes along the task (t) dimension
        self.prototypes = sub_prototypes if self.prototypes is None else torch.cat([self.prototypes, sub_prototypes], dim=1)
        
        # Calculate the squared differences for the current sub-tasks
        sub_squared_diff = (
                rearrange(train_x_enc, 'b (t s) d -> b t s d', t=self.config['sub_tasks']) -
                rearrange(sub_prototypes, 'b t d -> b t 1 d')
        ).square()
        sub_squared_diff = reduce(sub_squared_diff, 'b t s d -> b t d', 'sum')

        # Update the squared differences by concatenating the new sub_squared_diff along the task (t) dimension
        self.squared_diff = sub_squared_diff if self.squared_diff is None else torch.cat([self.squared_diff, sub_squared_diff], dim=1)

        alpha_prime = self.alpha + self.config['train_shots'] / 2
        beta_prime = self.beta + self.squared_diff / 2

        # The var of a class should actually stay the same unless we see new samples from that class.
        # Here, the var of new classes is added via the concatenation of the sub_squared_diff and self.squared_diff tensors.
        self.var = beta_prime / alpha_prime * (1 + 1 / self.config['train_shots']) 


    def test_forward(self, test_x: torch.Tensor, test_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_prime = self.alpha + self.config['train_shots'] / 2
        batch, test_num = test_y.shape
        test_x_enc = self.x_encoder(test_x) # shape(batch tasks shots) seq_len hidden_dim
        test_x_enc = rearrange(test_x_enc, '(batch tasks shots) seq_len hidden_dim -> batch (tasks shots) (seq_len hidden_dim)', batch=self.config['episode_bite'], tasks=self.config['sub_tasks'], shots=self.config['test_shots'])

        test_x_enc = rearrange(test_x_enc, 'b l h -> b l 1 h')

        # * Batsi: Setup chunking for prototypes and var.
        # * It is too large to fit in memory otherwise.
        # * Calculating the difference to every single prototype at once is too memory intensive.
        # * So we do it in chunks.
    
        # * very hacky way of doing this. sorry.

        num_classes = self.prototypes.shape[1]
        chunk_size = 25
        nll_chunks = []
        eps = 1e-8

        # prototypes = rearrange(self.prototypes, 'b t h -> b 1 t h')
        # squared_diff = (test_x_enc - prototypes).square()  # b l t h
        # var = rearrange(self.var, 'b t h -> b 1 t h')

        # nll = (squared_diff / (var * alpha_prime * 2) + 1).log() * (alpha_prime + 0.5) + (var + eps).log()
        # nll = reduce(nll, 'b l t h -> b l t', 'sum')

        for i in range(0, num_classes, chunk_size):
            prototypes_chunk = self.prototypes[:, i:i+chunk_size, :]
            var_chunk = self.var[:, i:i+chunk_size, :]
            var_chunk = rearrange(var_chunk, 'b t h -> b 1 t h')

            squared_diff = (test_x_enc - rearrange(prototypes_chunk, 'b t h -> b 1 t h')).square()  # b l t h

            # * Batsi: Calculating the nll is class independent hence we can do it in chunks.
            nll_chunk = (squared_diff / (var_chunk * alpha_prime * 2) + 1).log() * (alpha_prime + 0.5) + (var_chunk + eps).log()
            nll_chunk = reduce(nll_chunk, 'b l t h -> b l t', 'sum')
            nll_chunks.append(nll_chunk)

        nll = torch.cat(nll_chunks, dim=2)

        logit = -nll
        loss = self.ce(rearrange(logit, 'b l t -> (b l) t'), rearrange(test_y, 'b l -> (b l)'))
        loss = reduce(loss, '(b n) -> b', 'mean', b=batch, n=test_num)

        return loss, standard_classification_calculation(logit, test_y, self.config['test_shots'])
    
    def reset_class_statistics(self) -> None:
        self.prototypes = None
        self.var = None
        self.squared_diff = None

        
def standard_classification_calculation(
    logit: torch.Tensor, 
    test_y: torch.Tensor, 
    shots: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates both the overall episode accuracy and the per-class accuracy.

    This function assumes the input data is ordered by class (e.g., k shots of 
    class 0, then k shots of class 1, etc.). It reshapes the correctness 
    tensor to calculate the mean accuracy for each distinct class block.

    Args:
        logit (torch.Tensor): The raw model outputs with shape 
            (batch_size, sequence_length, num_classes).
        test_y (torch.Tensor): The target labels with shape 
            (batch_size, sequence_length).
        shots (int): The number of samples (k) per class (n-way-k-shot). 
            Used to reshape the sequence length into (n * k).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - episode_acc (torch.Tensor): Shape (batch_size,). 
              The mean accuracy over the entire episode.
            - class_acc (torch.Tensor): Shape (batch_size, n_ways). 
              The mean accuracy calculated separately for each class.
    """
    # Shape: (batch, total_length)
    correct = (logit.argmax(-1) == test_y).float()
    
    # Shape: (batch,)
    episode_acc = reduce(correct, 'b l -> b', 'mean')
    
    # We group the length (l) into (n classes * k shots) and average over k
    # Shape: (batch, n_ways)
    class_acc = reduce(correct, 'b (n k) -> b n', 'mean', k=shots)
    
    return episode_acc, class_acc