# Binarized Recurrent Neural Network Evolution

This project evolves a small binarized recurrent neural network over byte-tokenized text using a mutation-only genetic algorithm.

The CUDA kernel evaluates candidate models quickly by using packed `-1/+1` weights and XNOR-popcount dot products.

## Files

- `brnn_cuda.cu`
  - CUDA/PyTorch extension implementing the fused forward pass.
  - Supports a variable number of feedforward matrices.

- `model.py`
  - `ReferenceBRNN`: unpacked PyTorch implementation.
  - `KernelBRNN`: packed CUDA-backed implementation.
  - Packing/unpacking helpers.

- `evolve.py`
  - Training/evolution loop.
  - Loads text bytes, maps them to token IDs, mutates model bits, and keeps accepted mutations.

- `infer.py`
  - Loads a checkpoint.
  - Runs simple next-token inference with the reference model.
  - Accepts an optional prompt string.

## Model Structure

The recurrent state has fixed size `256`.

Each timestep:

```text
x = current 256-dimensional state

for layer in feedforward layers:
    x = sign(x @ ff[layer])

carry = x[0:128]
read  = x[128:256]

logits = read @ head / 16

next_state = concat(carry, embed[token])
```

The number of feedforward matrices is configurable.

Unpacked feedforward shape:

```text
ff: [num_ff, 256, 256]
```

Packed feedforward shape:

```text
ff_p: [num_ff, 8, 256]
```

Default:

```text
num_ff = 7
```

This corresponds to `ff0, ff1, ..., ff6`.

## Requirements

- Python 3
- PyTorch with CUDA support
- NVIDIA GPU with CUDA toolchain available
- A training text file

## Training Data

The training file is read as raw bytes.

Each distinct byte is mapped to a token ID. The model supports at most `128` distinct byte values.

Example:

```bash
python evolve.py --file ./training.txt
```

If your file contains more than `128` distinct byte values, training will stop with an error.

## Running Evolution

Basic run:

```bash
python evolve.py --file ./training.txt
```

Choose number of feedforward layers:

```bash
python evolve.py --file ./training.txt --num-ff 7
```

Use fewer or more feedforward layers:

```bash
python evolve.py --file ./training.txt --num-ff 3
python evolve.py --file ./training.txt --num-ff 12
```

Resume from checkpoint:

```bash
python evolve.py --file ./training.txt --resume --checkpoint-path ./checkpoint.pt
```

When resuming, the checkpoint's saved `num_ff` is used.

## Useful Arguments

```bash
python evolve.py \
  --file ./training.txt \
  --num-ff 7 \
  --print-every 100 \
  --checkpoint-every 10000 \
  --checkpoint-path ./checkpoint.pt \
  --csv-path ./data.csv
```

Arguments:

- `--file`
  - Training file path.
  - Default: `./training.txt`

- `--vocab-size`
  - Maximum allowed byte vocabulary size.
  - Default: `128`

- `--num-ff`
  - Number of `256x256` feedforward matrices.
  - Default: `7`

- `--print-every`
  - Print status every N accepted generations.
  - Default: `100`

- `--checkpoint-every`
  - Save checkpoint every N attempted mutation steps.
  - Default: `10000`

- `--checkpoint-path`
  - Checkpoint path.
  - Default: `./checkpoint.pt`

- `--csv-path`
  - CSV log path.
  - Default: `./data.csv`

- `--resume`
  - Resume from checkpoint.

- `--seed`
  - Random seed.
  - Default: `1`

- `--device`
  - CUDA device.
  - Default: `auto`

## Inference

After training has produced a checkpoint, run:

```bash
python infer.py --checkpoint-path ./checkpoint.pt
```

With a prompt:

```bash
python infer.py --checkpoint-path ./checkpoint.pt --prompt "hello"
```

The script loads the checkpoint into `ReferenceBRNN` and predicts the next token using greedy argmax.

## Checkpoints

Checkpoints contain:

- Unpacked reference model weights
- `num_ff`
- byte-to-token mapping
- token-to-byte mapping
- RNG states
- current step
- current generation

The model weights are saved unpacked for portability.

## Notes

- CUDA is required for `evolve.py`.
- `infer.py` uses `ReferenceBRNN`, so it can run on CPU if CUDA is unavailable.
- The model vocabulary is fixed at `128`.
- The hidden/state dimensions are fixed; only the number of feedforward layers is configurable.
