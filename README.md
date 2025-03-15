# Custom Fine-Tuning and Contrastive Learning with Transformers

## Running the Code
To run the code, specify the task by inputting the following in the terminal:

```bash
python3 main.py --task [baseline, custom1, custom2, supcon]
```

## Summary
In these experiments, we aim to understand the effects of different architectures and modeling techniques when using a pre-trained BERT encoder in our model to do scenario classification. We experiment with Stochastic Weight Averaging, Top Layer Reinitialization, Low-Rank Adaptation, and Contrastive Learning (SupCon and SimCLR). We found that using both Stochastic Weight Averaging and Top Layer Reinitialization resulted in the best validation accuracy of 91.15%. The accuracy for SupCon was very close and many of the other techniques were within 2% of this accuracy. SimCLR and LoRA performed the worst at about 81%.
