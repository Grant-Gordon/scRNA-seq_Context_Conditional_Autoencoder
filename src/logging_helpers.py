import os
import torch
from typing import Dict, Tuple, List, Any
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt



def init_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard_logs"))
    return writer 
##################LOSS LOGS############################
def per_chunk_raw_loss(
        writer: SummaryWriter,
        chunks_trained_on: int,
        chunk_loss_terms: Dict[str, float]
    ) -> None:
    """
    Logs a summary of total, reconstruction, integration, and Adverserial loss at the end of each chunk.
    """
    writer.add_scalars(f"loss_raw/chunk_raw_loss", {
        'total': chunk_loss_terms["aggreg"],
        'recon': chunk_loss_terms["recon"],
        'integ': chunk_loss_terms["integ"],#TODO: COMMENTED OUT FOR DEBUG ONLY
        'adv': chunk_loss_terms["adv_mean"] 
    }, chunks_trained_on)


def per_chunk_normed_loss(
        writer: SummaryWriter,
        chunks_trained_on: int,
        chunk_loss_terms: Dict[str, float]
    ) -> None:
    """
    Logs a summary of total, reconstruction, integration, and Adverserial loss at the end of each chunk.
    """
    writer.add_scalars(f"loss_normed/chunk_raw_loss", {
        'total': chunk_loss_terms["aggreg"],
        'recon': chunk_loss_terms["recon"],
        'integ': chunk_loss_terms["integ"], #TODO: COMMENTED OUT FOR DEBUG ONLY
        'adv': chunk_loss_terms["adv_mean"] 
    }, chunks_trained_on)

def per_chunk_adv_field_loss(
    writer: SummaryWriter,
    chunk_index: int,
    raw_adv_field_loss,
) -> None:
    """
    Log all adversarial field losses together on a single multi-line plot for this chunk. 
    """
    scalars = {}
    for field_name, value in raw_adv_field_loss.items():
        if torch.is_tensor(value):
            scalars[field_name] = float(value.detach().cpu())
        else:
            scalars[field_name] = float(value)

    # One tag → many lines inside one plot
    writer.add_scalars(
        "loss/adv_fields_raw/chunk",  # Plot group
        scalars,                      # {field_a: v, field_b: v, ...}
        global_step=chunk_index
    )



def per_epoch_raw_loss(
        writer: SummaryWriter,
        epoch: int,
        epoch_loss_terms: Dict[str, float]
    ) -> None:
    """
    Logs a summary of total, reconstruction, integration, and Adverserial loss at the end of each chunk.
    """
    writer.add_scalars(f"loss_raw/epoch_loss", {
        'total': epoch_loss_terms["aggreg"],
        'recon': epoch_loss_terms["recon"],
        'integ': epoch_loss_terms["integ"],#TODO: COMMENTED OUT FOR DEBUG ONLY
        'adv': epoch_loss_terms["adv_mean"] 
    }, epoch)

def per_epoch_normed_loss(
        writer: SummaryWriter,
        epoch: int,
        epoch_loss_terms: Dict[str, float]
    ) -> None:
    """
    Logs a summary of total, reconstruction, integration, and Adverserial loss at the end of each chunk.
    """
    writer.add_scalars(f"loss_normed/epoch_loss", {
        'total': epoch_loss_terms["aggreg"],
        'recon': epoch_loss_terms["recon"],
        'integ': epoch_loss_terms["integ"], #TODO: COMMENTED OUT FOR DEBUG ONLY
        'adv': epoch_loss_terms["adv_mean"] 
    }, epoch)

def per_epoch_raw_adv_field_loss(
    writer: SummaryWriter,
    epoch_index: int,
    raw_adv_field_loss,
) -> None:
    """
    Log all adversarial field losses together on a single multi-line plotfor this epoch.
    """
    scalars = {}
    for field_name, value in raw_adv_field_loss.items():
        if torch.is_tensor(value):
            scalars[field_name] = float(value.detach().cpu())
        else:
            scalars[field_name] = float(value)

    writer.add_scalars(
        "loss/adv_fields_raw/epoch",
        scalars,
        global_step=epoch_index
    )

def per_chunk_classifier_loss(
        writer: SummaryWriter,
        chunks_trained_on: int,
        chunk_classifier_loss: float
    )->None:
    writer.add_scalar("loss_raw/chunk_classifier_loss", chunk_classifier_loss, chunks_trained_on)

def per_epoch_classifier_loss(
        writer: SummaryWriter,
        epoch: int,
        epoch_classifier_loss: float
    )->None:
    writer.add_scalar("loss_raw/epoch_classifier_loss", epoch_classifier_loss, epoch)

###################Gradient Logs #############################
def per_chunk_grad_norms(writer, chunks_trained_on, grad_ems):
    writer.add_scalars(f"grad/chunk_gradient_norms", {
            'mean': grad_ems.get_Mean(),
            'min': grad_ems.get_Min(),
            'max': grad_ems.get_Max(),
        }, chunks_trained_on)
    
###################Gradient Logs #############################
def log_metadata_influence(
    writer: Any,
    metadata_ems: Dict[str, Any],
    global_step= None,
) -> None:
    """
    Log per-metadata-head influence statistics (min/mean/max L2) as a bar plot
    and as a markdown table to TensorBoard.

    Parameters
    ----------
    writer : SummaryWriter-like
        TensorBoard writer with `add_figure` and `add_text` methods.
    metadata_ems : Dict[str, Ems]
        Mapping from head name to an Ems instance tracking L2 stats.
    global_step : Optional[int]
        Global step / epoch index for logging. If None, defaults to 0.
    """
    if global_step is None:
        global_step = 0

    head_names = []
    means = []
    mins = []
    maxs = []

    # Collect stats
    for head, ems in metadata_ems.items():
        head_names.append(head)

        # Adjust these getters to match your actual Ems API
        mean_val = ems.get_Mean()
        min_val = ems.get_Min()
        max_val = ems.get_Max()

        # Convert None to NaN so numpy / plotting can handle it
        mean_val = np.nan if mean_val is None else float(mean_val)
        min_val = np.nan if min_val is None else float(min_val)
        max_val = np.nan if max_val is None else float(max_val)

        means.append(mean_val)
        mins.append(min_val)
        maxs.append(max_val)

    if not head_names:
        # Nothing to log
        return

    means_arr = np.array(means, dtype=float)
    mins_arr = np.array(mins, dtype=float)
    maxs_arr = np.array(maxs, dtype=float)

    num_heads = len(head_names)
    x = np.arange(num_heads)
    width = 0.25

    # Figure size scales with number of heads
    fig, ax = plt.subplots(
        figsize=(max(6.0, num_heads * 0.5), 4.0)
    )

    offset_idx = 0

    # Min
    ax.bar(x + (offset_idx - 1) * width, mins_arr, width, label="min")
    offset_idx += 1

    # Mean
    ax.bar(x + (offset_idx - 1) * width, means_arr, width, label="mean")
    offset_idx += 1

    # Max
    ax.bar(x + (offset_idx - 1) * width, maxs_arr, width, label="max")
    offset_idx += 1

    ax.set_xticks(x)
    ax.set_xticklabels(head_names, rotation=45, ha="right")
    ax.set_ylabel("L2 magnitude of metadata offset")
    ax.set_title("Metadata head influence (L2 of C_t)")
    ax.legend()

    fig.tight_layout()
    writer.add_figure("metadata/head_influence_l2", fig, global_step=global_step)
    plt.close(fig)

    # Also log a simple text table for readability
    lines = [
        "| Head | Mean L2 | Min L2 | Max L2 |",
        "|------|---------|--------|--------|",
    ]
    for head, mean_val, min_val, max_val in zip(head_names, means_arr, mins_arr, maxs_arr):
        mean_str = f"{mean_val:.4f}" if np.isfinite(mean_val) else "NA"
        min_str = f"{min_val:.4f}" if np.isfinite(min_val) else "NA"
        max_str = f"{max_val:.4f}" if np.isfinite(max_val) else "NA"
        lines.append(f"| {head} | {mean_str} | {min_str} | {max_str} |")

    table_text = "\n".join(lines)
    writer.add_text("metadata/head_influence_l2_stats", table_text, global_step=global_step)

class Ems:
    def __init__(self, use_max=True, use_min=True, use_mean=True, use_mode=False, use_median=False):
        self.use_max = use_max
        self.use_min = use_min
        self.use_mean = use_mean
        self.use_mode = use_mode
        self.use_median = use_median 
        if use_max: self.Max = None
        if use_min: self.Min = None
        if use_mean: 
            self.Mean = None
            self.sum = None
            self.count =None
        if use_mode: 
            self.Mode = None
            self.occurances = {}
        if use_median: 
            self.Median=None
            self.elements = []

    def update(self, val):
        if val ==  None:
            return
        
        if self.use_max and (self.Max == None or val > self.Max):
            self.Max =val

        if self.use_min and (self.Min == None or val < self.Min):
            self.Min = val
        
        if self.use_mean:
            if self.sum is not None:
                self.sum+=val
            else:
                self.sum = val
            if self.count is not None:
                self.count+=1
            else:
                self.count=1

        if self.use_mode:
            if val in self.occurances:
                self.occurances[val]+=1
            else:
                self.occurances[val] = 1

        if self.use_median:
            self.elements.append(val)

    def get_Max(self):
        return self.Max
    def set_Max(self, max):
        self.Max = max
    def get_Min(self):
        return self.Min
    def set_Min(self, min):
        self.Min = min
    def get_Mean(self):
        if self.count != 0:
            return self.sum / self.count
        else: 
            return None

    def set_Mean(self, mean, sum, count):
        self.Mean = mean
        self.sum = sum
        self.count = count
    def get_Mode(self):
        key = max(self.occurances, key=self.occurances.get)
        return key, self.occurances[key]
    def set_Mode(self, mode, occurances):
        self.Mode = mode
        self.occurances = occurances
