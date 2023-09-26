import torch

rand_var = 123
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"