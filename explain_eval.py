import argparse
import torch
import random
import numpy as np
from explainability import eval_pos_explain, explain_eval_fid
import warnings

def write_results(lst, file):
    # open file
    with open(file, 'w+') as f:
        # write elements of list
        for items in lst:
            f.write('%s\n' %items)
        print("File written successfully")
    # close the file
    f.close()    

def main():
    warnings.filterwarnings("ignore")
    
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Process some parameters.')

    # Add --dataset argument with choices
    parser.add_argument('--dataset', type=str, choices=['bitcoinotc', 'reddit-title','email-eu'], required=True,
                        help='Choose the dataset from: [bitcoinotc, reddit-title]')

    # Add --model argument with choices
    parser.add_argument('--model', type=str, choices=['evolvegcn', 'gcrngru', 'roland'], required=True,
                        help='Choose the model from: [evolvegcn, gcrngru, roland]')

    parser.add_argument('--xai_model', type=str, choices=['gnnexplainer', 'last', 'khop', 'sa', 'ig', 'dummy', 'pg'], required=True,
                        help='Choose the XAI model \
                        [gnnexplainer, last (LastSnapshotExplainer), khop (2-hop temporal subgraph explainer),\
                        sa (Saliency maps), ig (IntegratedGradients), dummy (random explainer), pg (PGExplainer)')

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
    xai_str = args.xai_model

    # Print out the chosen options
    print(f"Model to explain: {args.model}")
    print(f"Dataset to explain: {args.dataset}")
    print(f"Explanations provided by: {args.xai_model}")
    print(f"Assuming {model_str} is already trained and saved as trained_models/{model_str}_{dataset_str}.pt")

    coh, recurrence, reciprocity, homophily , num_recs, num_recips = eval_pos_explain(model_str, dataset_str, xai_str = xai_str, device=device)

    write_results(coh, f'logs/xai_results/{dataset_str}/{model_str}/{xai_str}_coh.txt')
    write_results(recurrence, f'logs/xai_results/{dataset_str}/{model_str}/{xai_str}_rec.txt')
    write_results(reciprocity, f'logs/xai_results/{dataset_str}/{model_str}/{xai_str}_recip.txt')
    write_results(homophily, f'logs/xai_results/{dataset_str}/{model_str}/{xai_str}_hom.txt')
    write_results([num_recs], f'logs/xai_results/{dataset_str}/{model_str}/{xai_str}_num_rec.txt')
    write_results([num_recips], f'logs/xai_results/{dataset_str}/{model_str}/{xai_str}_num_recip.txt')
    
if __name__ == '__main__':
    main()