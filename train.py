import argparse
import torch
import random
import numpy as np
from explainability import eval_pos_explain, explain_eval_fid
import warnings
from train_models import train_gcrngru, train_evolvegcn, train_roland
from models import save_trained_model


def main():
    warnings.filterwarnings("ignore")
    
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Process some parameters.')

    # Add --dataset argument with choices
    parser.add_argument('--dataset', type=str, choices=['bitcoinotc', 'reddit-title','email-eu'], required=True,
                        help='Choose the dataset from: [bitcoinotc, reddit-title, email-eu]')

    # Add --model argument with choices
    parser.add_argument('--model', type=str, choices=['evolvegcn', 'gcrngru', 'roland'], required=True,
                        help='Choose the model from: [evolvegcn, gcrngru, roland]')

    parser.add_argument('--seed', type=int, help='Random seed', default=1234)

    # Parse the arguments
    args = parser.parse_args()

    seed = args.seed
    device = torch.device('cuda')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()

    model_str = args.model
    dataset_str = args.dataset

    # Print out the chosen options
    print(f"Model to train: {args.model}")
    print(f"Dataset: {args.dataset}")

    if model_str == 'evolvegcn':
        model, _ = train_evolvegcn(model_str, dataset_str, device=device)
    elif model_str == 'gcnrngru':
        model, _ = train_gcrngru(model_str, dataset_str, device=device)
    elif model_str == 'roland':
        model, _ = train_roland(model_str, dataset_str, device=device)

    print(f"Saving {model} as trained_models/{model_str}_{dataset_str}.pt")

    save_trained_model(model, model_str, dataset_str)
    
    print('Done')
    
if __name__ == '__main__':
    main()