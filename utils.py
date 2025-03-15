import os
import numpy as np
import torch
import random
import re
import matplotlib.pyplot as plt



def check_directories(args):
    task_path = os.path.join(args.output_dir)
    if not os.path.exists(task_path):
        os.mkdir(task_path)
        print(f"Created {task_path} directory")
    
    folder = args.task
    
    save_path = os.path.join(task_path, folder)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print(f"Created {save_path} directory")
    args.save_dir = save_path

    cache_path = os.path.join(args.input_dir, 'cache')
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
        print(f"Created {cache_path} directory")

    if args.debug:
        args.log_interval /= 10

    return args

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def setup_gpus(args):
    n_gpu = 0  # set the default to 0
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
    args.n_gpu = n_gpu
    if n_gpu > 0:   # this is not an 'else' statement and cannot be combined
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return args


def plot_losses(train_losses, val_losses, args, best_epoch=None):

    # Create 'plots' directory if it doesn't exist
    if not os.path.isdir('results'):
        os.mkdir('results')

    # Plotting training and validation losses
    plt.plot(train_losses, label='Training Accuracy', linestyle='-', marker=None)
    plt.plot(val_losses, label='Validation Accuracy', linestyle='-', marker=None)

    # If best_epoch is provided, mark it on the validation loss curve with an "X"
    if best_epoch is not None and 0 <= best_epoch < len(val_losses):
        plt.scatter(best_epoch, val_losses[best_epoch], color='r', marker='x', s=100, label=f'Best Model (Epoch {best_epoch})')

    # Adjusts to the model task
    if args.task == 'lora':
        fname=f"Lora Model with Rank {args.rank}"
    elif args.task == 'baseline':
        fname=f"Baseline Model"
    elif args.task == 'custom1':
        fname="SWA Model"
    elif args.task == 'custom2':
        fname=f"Reinitialization of the Top {args.reinit_layers} Layers"
    elif args.task == 'custom':
        fname=f"SWA Model with Reinitialization of the Top {args.reinit_layers} Layers"
    else:
        fname=f"{args.task.capitalize()} Model"

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f"{fname}: Accuracy per Epoch")
    plt.legend()

    # Saving the plot as an image file in 'plots' directory
    save_spot = f"results/{args.task}/{fname}.png"
    plt.savefig(save_spot)
    print(f"Plot saved as: {save_spot}")
