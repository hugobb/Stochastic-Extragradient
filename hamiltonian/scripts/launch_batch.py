import submitit
from hamiltonian.games import GaussianBilinearGame
from hamiltonian.algorithms import ConsensusOptimization
import torch
import argparse
from enum import Enum


class Game(Enum):
    BILINEAR = "bilinear"


class Optimizer(Enum):
    CONSENSUS_OPTIMIZATION = "consensus_optimization"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=str) 
    parser.add_argument("--n_samples", default=100, type=int)
    parser.add_argument("--dim", default=100, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--game", default=Game.BILINEAR, choices=Game, type=Game)
    parser.add_argument("--game", default=Optimizer.CONSENSUS_OPTIMIZATION, choices=Optimizer, type=Optimizer)
    parser.add_argument("--lr", default=0.1, default=float)
    parser.add_argument("--lr_H", default=0.1, default=float)
    parser.add_argument("--n_runs", default=5, default=int)
    parser.add_argument("--n_iter", default=1000000, default=int)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    if args.game == Game.BILINEAR:
        game = GaussianBilinearGame(args.n_samples, args.dim, bias=True)
    else:
        raise NotImplementedError()

    if args.optimizer == Optimizer.CONSENSUS_OPTIMIZATION:
        optimizer = ConsensusOptimization(game, lr=lr, lr_H=lr_H, full_batch=False)

    log_folder = "log_test/%j"
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(timeout_min=4, slurm_partition="dev")
    jobs = []
    with executor.batch():
        for i in range(args.num_runs):
            job = executor.submit(optimizer.run, args.n_iter)
            jobs.append(job)


