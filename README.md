Current config for 8 GPUs with 32 mini-batch size each.

The layer names are designed to match MSRA released pre-trained models to allow for finetuning.

Running the ResNet-50 as-is gets a few percent lower accuracy than MSRA if done without random reshape & crop. 
