# Genetic Algorithms

This subfolder is not necessary for the overall BRNN on C64 project; it experiments with genetic algorithms for training binarized RNNs.

This folder contains three subprojects: 256, 512, and 512T.

Each one has the same structure of the following files:

```
brnn_cuda.cu  # cuda kernel for fast loss evaluation
model.py  # defines model; both a pytorch version and an optimized CUDA version
infer.py  # generation script
evolve.py  # training script based on simple mutation accept/reject loop
plot_progress.py  # graphical plot of progress over time
```

In each case, the code defines a binarized RNN which can be quickly evaluated on data using a CUDA kernel. The number 256 vs 256 is whether the feedforward matrix is 256x256 or 512x512; the T represents that thresholds are used.

Note: 256T wasn't implemented before this approach to training Binarized RNN was abandoned for QAT, so it isn't here.
