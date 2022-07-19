from functools import partial
from npf import CNP
from npf.architectures import MLP, merge_flat_input
from utils.helpers import count_parameters
from utils.train import train_model
from npf import CNPFLoss

R_DIM = 128
KWARGS = dict(
    XEncoder=partial(MLP, n_hidden_layers=1, hidden_size=R_DIM),
    Decoder=merge_flat_input(  # MLP takes single input but we give x and R so merge them
        partial(MLP, n_hidden_layers=4, hidden_size=R_DIM), is_sum_merge=True,
    ),
    r_dim=R_DIM,
)

# 1D case
model_1d = partial(
    CNP,
    x_dim=1,
    y_dim=1,
    XYEncoder=merge_flat_input(  # MLP takes single input but we give x and y so merge them
        partial(MLP, n_hidden_layers=2, hidden_size=R_DIM * 2), is_sum_merge=True,
    ),
    **KWARGS,
)

n_params_1d = count_parameters(model_1d())
print(f"Number Parameters (1D): {n_params_1d:,d}")

KWARGS = dict(
    is_retrain=False,  # whether to load precomputed model or retrain
    criterion=CNPFLoss,  # Standard loss for conditional NPFs
    chckpnt_dirname="results/pretrained/",
    device=None,  # use GPU if available
    batch_size=32,
    lr=1e-3,
    decay_lr=10,  # decrease learning rate by 10 during training
    seed=123,
)

# 1D
trainers_1d = train_model(
    gp_datasets,
    {"CNP": model_1d},
    test_datasets=gp_test_datasets,
    train_split=None,  # No need of validation as the training data is generated on the fly
    iterator_train__collate_fn=context_set,
    iterator_valid__collate_fn=context_set,
    max_epochs=100,
    **KWARGS
)

