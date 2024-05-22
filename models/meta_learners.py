import pytorch_lightning as pl
import torch
import torch.nn as nn
import utils.utils as utils
from torch.utils.data import DataLoader


# MLP for multi-class classification (discrete conditional distributions) or continuous regression

