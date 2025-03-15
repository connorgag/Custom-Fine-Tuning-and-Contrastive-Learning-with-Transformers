import os, sys, pdb
import numpy as np
import random
import torch

import math

from tqdm import tqdm as progress_bar

from torch.optim.swa_utils import AveragedModel, SWALR

from utils import set_seed, setup_gpus, check_directories, plot_losses
from dataloader import get_dataloader, check_cache, prepare_features, process_data, prepare_inputs
from load import load_data, load_tokenizer
from arguments import params
from model import ScenarioModel, SupConModel, CustomModel, ClassifierWrapper
from torch import nn
from loss import SupConLoss

from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def baseline_train(args, model, datasets, tokenizer):


    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, datasets["train"], split = "train")
    training_acc = []
    valid_accu = []
    best_model_acc = 0
    best_model_epoch = 0
    

    # task2: setup model's optimizer_scheduler if you have

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = CosineAnnealingLR(
            optimizer, 
            T_max = args.n_epochs,
            eta_min=1e-6 
        )
    
    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()
        acc = 0

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch, use_text = False)
            logits = model(inputs, labels)
            tem = (logits.argmax(1) == labels).float().sum()
            acc += tem.item()
            
            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()  # backprop to update the weights
            scheduler.step()  # Update learning rate schedule
            optimizer.zero_grad()
            losses += loss.item()

            
        
    
        valid_acc = run_eval(args, model, datasets, tokenizer, split='validation')
        
        valid_accu.append(valid_acc)
        training_acc.append(acc/len(datasets["train"]))
                            
        print('epoch', epoch_count, '| losses:', losses)

        if (acc/len(datasets["train"])) > best_model_acc:
                best_model_acc = acc/len(datasets["train"])
                best_model_epoch = epoch_count

    print("Training Accuracy: ", training_acc[-1])
    print("Validation Accuracy: ", valid_accu[-1])

    plot_losses(training_acc, valid_accu, args, best_epoch=best_model_epoch)


def custom_train(args, model, datasets, tokenizer):
    """
    Custom training loop with SWA.
    """

    criterion = nn.CrossEntropyLoss()

    train_dataloader = get_dataloader(args, datasets["train"], split="train")

    training_acc = []
    valid_accu = []
    best_model_acc = 0
    best_model_epoch = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=1e-6)

    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=3*args.learning_rate)

    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()
        acc = 0

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch, use_text=False)
            logits = model(inputs, labels)
            tem = (logits.argmax(1) == labels).float().sum()
            acc += tem.item()

            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            losses += loss.item()

        if epoch_count < (0.75 * args.n_epochs):
            scheduler.step()
        else:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        valid_acc = run_eval(args, model, datasets, tokenizer, split='validation')

        valid_accu.append(valid_acc)
        training_acc.append(acc / len(datasets["train"]))

        print('epoch', epoch_count, '| losses:', losses)

        if (acc / len(datasets["train"])) > best_model_acc:
            best_model_acc = acc / len(datasets["train"])
            best_model_epoch = epoch_count

    torch.optim.swa_utils.update_bn(train_dataloader, swa_model)

    print("Training Accuracy: ", training_acc[-1])
    

    eval_acc = run_eval(args, swa_model, datasets, tokenizer, split='validation')

    print("Validation Accuracy: ", eval_acc)

    plot_losses(training_acc, valid_accu, args, best_epoch=best_model_epoch)

    return swa_model


def run_eval(args, model, datasets, tokenizer, split='validation'):
    model.eval()
    dataloader = get_dataloader(args, datasets[split], split)
    criterion = torch.nn.CrossEntropyLoss() 
    total_loss = 0

    acc = 0
    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch)
        logits = model(inputs, labels)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        
        tem = (logits.argmax(1) == labels).float().sum()
        acc += tem.item()

    avg_loss = total_loss / len(dataloader)
  
    print(f'{split} acc:', acc/len(datasets[split]), f'| {split} loss:', avg_loss, f'|dataset split {split} size:', len(datasets[split]))

    return acc/len(datasets[split])


def supcon_train(args, model, datasets, tokenizer):
    from loss import SupConLoss
    # model.train()
    criterion = SupConLoss()

    # task1: load training split of the dataset
    
    # task2: setup optimizer_scheduler in your model

    # task3: write a training loop for SupConLoss function 

    # if args.method == 'SupCon':
    #         loss = criterion(features, labels)
    # elif args.method == 'SimCLR':
    #         loss = criterion(features)
    # else:
    criterion_baseline = nn.CrossEntropyLoss()
    criterion=SupConLoss()
    train_dataloader = get_dataloader(args, datasets["train"], split = "train")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(
            optimizer, 
            T_max = args.n_epochs,
            eta_min=1e-6 
        )
    
    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            # print(batch)
            inputs, labels = prepare_inputs(batch, use_text = False)
            # print(done)
            # inputs=torch.cat(**inputs,**inputs,dim=0)
            concatenated = {
    'input_ids': torch.cat((inputs['input_ids'],inputs['input_ids']), dim=0),
    'token_type_ids': torch.cat((inputs['token_type_ids'],inputs['token_type_ids']), dim=0),
    'attention_mask': torch.cat((inputs['attention_mask'],inputs['attention_mask']), dim=0)
}
            bsz = labels.shape[0]
            features = model(concatenated, labels)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            if args.method == 'supcon':
                loss = criterion(features, labels)
            elif args.method == 'simclr':
                    loss = criterion(features)
            else:
              loss = criterion_baseline(features, labels)
            loss.backward()

            optimizer.step()  
            scheduler.step()  
            optimizer.zero_grad()
            losses += loss.item()
    
        run_eval(args, model, datasets, tokenizer, split='validation')
        print('epoch', epoch_count, '| losses:', losses)
    print("Finished contrastive training. Freezing base model...")
    for param in model.parameters():
        param.requires_grad = False

    feature_dim = features.size(-1)  
    classifier = ClassifierWrapper(model, nn.Linear(feature_dim, 60)).to(device)

    cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate_2)
    valid_accu=[]
    training_acc=[]
    best_model_acc = 0
    best_model_epoch = 0
    print("Training classifier...")
    for epoch_count in range(args.n_epochs_2):
        losses = 0
        classifier.train()
        acc=0

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch, use_text = False)
            logits = classifier(inputs, labels)
            tem = (logits.argmax(1) == labels).float().sum()
            acc += tem.item()
            loss = criterion_baseline(logits, labels)
            
            cls_optimizer.zero_grad()
            loss.backward()
            cls_optimizer.step()
            
            loss += loss.item()
    
        # run_eval()
        # print('epoch', epoch_count, '| losses:', losses)
        valid_acc = run_eval(args, classifier, datasets, tokenizer, split='validation')
        
        valid_accu.append(valid_acc)
        training_acc.append(acc/len(datasets["train"]))
                            
        print('epoch', epoch_count, '| losses:', losses)

        if (acc/len(datasets["train"])) > best_model_acc:
                best_model_acc = acc/len(datasets["train"])
                best_model_epoch = epoch_count
    plot_losses(training_acc, valid_accu, args, best_epoch=best_model_epoch)
        


if __name__ == "__main__":
  args = params()
  args = setup_gpus(args)
  args = check_directories(args)
  set_seed(args)

  cache_results, already_exist = check_cache(args)
  tokenizer = load_tokenizer(args)

  if already_exist:
    features = cache_results
  else:
    data = load_data()
    features = prepare_features(args, data, tokenizer, cache_results)
  datasets = process_data(args, features, tokenizer)
  # for k,v in datasets.items():
  #   print(k, len(v))
  
 
  if args.task == 'baseline':
    model = ScenarioModel(args, tokenizer, target_size=60).to(device)
    print("1st eval")
    run_eval(args, model, datasets, tokenizer, split='validation')
    print("2nd eval")
    run_eval(args, model, datasets, tokenizer, split='test')
    baseline_train(args, model, datasets, tokenizer)
    run_eval(args, model, datasets, tokenizer, split='test')

  elif args.task == 'lora':
    print(f"\nLoRA with rank {args.rank}")

    model = ScenarioModel(args, tokenizer, target_size=60, lora=True).to(device)

    print("Accuracy on validation")
    run_eval(args, model, datasets, tokenizer, split='validation')
    print("Accuracy on test split")
    run_eval(args, model, datasets, tokenizer, split='test')

    print("Training LoRA")
    baseline_train(args, model, datasets, tokenizer)
    print("Test split accuracy after training:")
    run_eval(args, model, datasets, tokenizer, split='test')

  elif args.task == 'custom1':
    model = ScenarioModel(args, tokenizer, target_size=60).to(device)
    run_eval(args, model, datasets, tokenizer, split='validation')
    run_eval(args, model, datasets, tokenizer, split='test')
    custom_train(args, model, datasets, tokenizer)
    run_eval(args, model, datasets, tokenizer, split='test')

  elif args.task == 'custom2':
    model = CustomModel(args, tokenizer, target_size=60, n_layers_to_reinitialize=args.reinit_layers).to(device)
    run_eval(args, model, datasets, tokenizer, split='validation')
    run_eval(args, model, datasets, tokenizer, split='test')
    baseline_train(args, model, datasets, tokenizer)
    run_eval(args, model, datasets, tokenizer, split='test')

  elif args.task == 'custom':
    model = CustomModel(args, tokenizer, target_size=60, n_layers_to_reinitialize=args.reinit_layers).to(device)
    run_eval(args, model, datasets, tokenizer, split='validation')
    run_eval(args, model, datasets, tokenizer, split='test')
    custom_train(args, model, datasets, tokenizer)
    run_eval(args, model, datasets, tokenizer, split='test')

  elif args.task == 'supcon':
    model = SupConModel(args, tokenizer, target_size=60).to(device)
    supcon_train(args, model, datasets, tokenizer)
