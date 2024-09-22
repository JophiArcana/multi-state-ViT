from argparse import Namespace

import numpy as np
import torch
from dimarray import DimArray
from tensordict import TensorDict

from infrastructure import utils
from infrastructure.settings import DEVICE
from system.linear_time_invariant import LTISystem


"""
Loading and generating args
"""
BaseDatasetArgs = Namespace(
    n_systems=Namespace(train=1),
    dataset_size=Namespace(train=1, valid=100, test=500),
    total_sequence_length=Namespace(train=2000, valid=20000, test=800000),
)
BaseTrainArgs = Namespace(
    # Batch sampling
    sampling=Namespace(
        method="subsequence_padded",        # {"full", "subsequence_padded", "subsequence_unpadded"}
        batch_size=128,
        subsequence_length=16
    ),

    # Optimizer
    optimizer=Namespace(
        type="AdamW",                        # {"SGD", "AdamW"}
        max_lr=2e-2, min_lr=1e-6,
        weight_decay=0.0,

        momentum=0.9,                       # SECTION: Used for Adam and SGD
    ),

    # Scheduler
    scheduler=Namespace(
        type="exponential",                 # {"exponential", "cosine"}
        warmup_duration=100,

        epochs=2500, lr_decay=0.995,        # SECTION: Used for exponential scheduler
        T_0=10, T_mult=2, num_restarts=8,   # SECTION: Used for cosine scheduler
    ),

    # Iteration
    iterations_per_epoch=20,

    # Loss
    control_coefficient=1.0,
)
BaseExperimentArgs = Namespace(
    n_experiments=1,
    ensemble_size=32,
    backup_frequency=None
)

def args_from(HP: Namespace):
    HP.system = utils.process_defaulting_roots(HP.system)
    HP.dataset = utils.process_defaulting_roots(HP.dataset)
    return HP

def load_system_and_args(folder: str):
    A = torch.Tensor(np.loadtxt(f"{folder}/A.out", delimiter=",")).to(DEVICE)
    B = torch.Tensor(np.loadtxt(f"{folder}/B.out", delimiter=","))[:, None].to(DEVICE)
    C = torch.Tensor(np.loadtxt(f"{folder}/C.out", delimiter=","))[None].to(DEVICE)
    input_enabled = not bool(torch.all(torch.isclose(B, torch.zeros_like(B))))

    S_D = A.shape[0]
    O_D = C.shape[0]
    I_D = B.shape[1]

    noise_block = torch.Tensor(np.loadtxt(f"{folder}/noise_block.out", delimiter=",")).to(DEVICE)
    W = noise_block[:S_D, :S_D]
    V = noise_block[S_D:, S_D:]

    sqrt_W, sqrt_V = utils.sqrtm(W), utils.sqrtm(V)

    problem_shape = Namespace(
        environment=Namespace(observation=O_D),
        controller=Namespace(input=I_D) if input_enabled else Namespace(),
    )
    auxiliary = Namespace()
    args = utils.deepcopy_namespace(Namespace(
        system=Namespace(
            S_D=S_D, problem_shape=problem_shape,
            auxiliary=auxiliary,
        ),
        dataset=BaseDatasetArgs,
        model=Namespace(problem_shape=problem_shape),
        training=BaseTrainArgs,
        experiment=BaseExperimentArgs
    ))
    args.dataset.n_systems.train = 1
    args.experiment.n_experiments = 1

    system_group = LTISystem(problem_shape, auxiliary, TensorDict.from_dict({"environment": {
        "F": A, "B": TensorDict({"input": B}, batch_size=()), "H": C, "sqrt_S_W": sqrt_W, "sqrt_S_V": sqrt_V
    }}, batch_size=()).expand(args.dataset.n_systems.train, args.experiment.n_experiments))
    return {"train": DimArray(utils.array_of(system_group), dims=[])}, args_from(args)

def generate_args(shp: Namespace) -> Namespace:
    return args_from(utils.deepcopy_namespace(Namespace(
        system=shp,
        dataset=BaseDatasetArgs,
        model=Namespace(problem_shape=shp.problem_shape),
        training=BaseTrainArgs,
        experiment=BaseExperimentArgs
    )))




