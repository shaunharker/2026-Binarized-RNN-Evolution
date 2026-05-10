# evolve.py

import argparse
import csv
import math
import os
import random
from datetime import datetime
from typing import Dict, Tuple

import torch

from model import ReferenceBRNN, KernelBRNN, DEFAULT_NUM_FF


MODEL_VOCAB_SIZE = 128
SEQUENCE_LENGTH = 128
BATCH_TOKENS = 2**20
BATCH_SIZE = BATCH_TOKENS // SEQUENCE_LENGTH


# ============================================================
# Data Loading
# ============================================================

def load_byte_training_file(
    path: str,
    vocab_size: int,
) -> Tuple[torch.Tensor, Dict[int, int], Dict[int, int]]:
    with open(path, "rb") as f:
        raw = f.read()

    if len(raw) == 0:
        raise ValueError(f"{path} is empty")

    distinct = sorted(set(raw))

    if len(distinct) > vocab_size:
        raise ValueError(
            f"File contains {len(distinct)} distinct byte values, "
            f"but vocab_size is {vocab_size}. Increase --vocab-size."
        )

    if vocab_size > MODEL_VOCAB_SIZE:
        raise ValueError(
            f"vocab_size must be <= {MODEL_VOCAB_SIZE}; "
            f"the BRNN head/embed size is fixed at {MODEL_VOCAB_SIZE}."
        )

    byte_to_id = {b: i for i, b in enumerate(distinct)}
    id_to_byte = {i: b for b, i in byte_to_id.items()}

    encoded = torch.tensor(
        [byte_to_id[b] for b in raw],
        dtype=torch.long,
    )

    return encoded, byte_to_id, id_to_byte


def make_training_batch(encoded_cpu: torch.Tensor, device: torch.device) -> torch.Tensor:
    if encoded_cpu.numel() < BATCH_TOKENS:
        raise ValueError(
            f"Training file contains only {encoded_cpu.numel()} encoded tokens, "
            f"but this script needs at least {BATCH_TOKENS} tokens to form "
            f"a [{BATCH_SIZE}, {SEQUENCE_LENGTH}] batch."
        )

    return (
        encoded_cpu[:BATCH_TOKENS]
        .to(device=device, dtype=torch.long)
        .view(BATCH_SIZE, SEQUENCE_LENGTH)
        .contiguous()
    )


# ============================================================
# Device
# ============================================================

def resolve_device(device_arg: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because evolve.py uses KernelBRNN.")

    if device_arg == "auto":
        device = torch.device("cuda")
    else:
        device = torch.device(device_arg)

    if device.type != "cuda":
        raise RuntimeError("KernelBRNN requires a CUDA device. Use --device cuda or --device cuda:N.")

    if device.index is not None:
        torch.cuda.set_device(device)

    return torch.device("cuda", torch.cuda.current_device())


# ============================================================
# Checkpoint Saving and Loading
# ============================================================

def save_checkpoint(
    path: str,
    model: KernelBRNN,
    byte_to_id: Dict[int, int],
    id_to_byte: Dict[int, int],
    step: int,
    generation: int,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Save an unpacked ReferenceBRNN state_dict on CPU for portability.
    ref_model = ReferenceBRNN(model)
    state_dict_cpu = {
        key: value.detach().cpu()
        for key, value in ref_model.state_dict().items()
    }

    payload = {
        "step": int(step),
        "generation": int(generation),
        "num_ff": int(model.num_ff),
        "model_state_dict": state_dict_cpu,
        "byte_to_id": byte_to_id,
        "id_to_byte": id_to_byte,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "python_random_state": random.getstate(),
    }

    torch.save(payload, path)

    now = datetime.now()
    print(f"\n>>> Checkpoint saved to {path} at step {step}")
    print(f">>> Time: {now.strftime('%Y-%m-%d %H:%M:%S')} ({int(now.timestamp())})")


def torch_load_checkpoint(path: str):
    """
    Compatibility wrapper.

    Newer PyTorch versions support weights_only. Older versions do not.
    This checkpoint contains Python RNG state, so weights_only=False is needed
    when the argument exists.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_checkpoint(
    path: str,
    device: torch.device,
) -> Tuple[KernelBRNN, int, int, Dict[int, int], Dict[int, int]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    print(f"Loading checkpoint: {path}")
    checkpoint = torch_load_checkpoint(path)

    state_dict = checkpoint["model_state_dict"]

    if "num_ff" in checkpoint:
        num_ff = int(checkpoint["num_ff"])
    elif "ff" in state_dict:
        num_ff = int(state_dict["ff"].shape[0])
    else:
        num_ff = DEFAULT_NUM_FF

    ref_model = ReferenceBRNN(num_ff=num_ff)
    ref_model.load_state_dict(state_dict)

    model = KernelBRNN(ref_model)
    model.eval()

    step = int(checkpoint.get("step", 0))
    generation = int(checkpoint.get("generation", 0))

    byte_to_id = checkpoint.get("byte_to_id", {})
    id_to_byte = checkpoint.get("id_to_byte", {})

    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])

    if "cuda_rng_state_all" in checkpoint:
        cuda_states = checkpoint["cuda_rng_state_all"]
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)

    if "python_random_state" in checkpoint:
        random.setstate(checkpoint["python_random_state"])

    print(f"Resumed from step {step}, generation {generation}, num_ff {model.num_ff}.")
    return model, step, generation, byte_to_id, id_to_byte


# ============================================================
# Genetic Algorithm
# ============================================================

@torch.no_grad()
def objective_loss(model: KernelBRNN, batch: torch.Tensor) -> float:
    """
    Returns mean negative log-likelihood in nats per token.

    KernelBRNN.loss returns one accumulated sequence loss per batch element,
    so divide by sequence length after averaging over the batch.
    """
    return float(model(batch).mean().div(batch.shape[1]).item())


@torch.no_grad()
def genetic_alg_step(model: KernelBRNN, batch: torch.Tensor) -> Tuple[float, int]:
    loss = objective_loss(model, batch)
    tries = 0

    while True:
        tries += 1

        target, index = model.mutate()
        mutant_loss = objective_loss(model, batch)

        if mutant_loss < loss:
            return mutant_loss, tries

        # Revert rejected mutation.
        model.mutate(target, index)


# ============================================================
# Logging
# ============================================================

def append_csv_row(
    csv_path: str,
    generation: int,
    step: int,
    loss_nats_per_token: float,
) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    file_exists = os.path.isfile(csv_path)
    loss_bits_per_token = loss_nats_per_token / math.log(2.0)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(
                [
                    "generation",
                    "step",
                    "loss_nats_per_token",
                    "loss_bits_per_token",
                    "unixtime",
                ]
            )

        writer.writerow(
            [
                generation,
                step,
                loss_nats_per_token,
                loss_bits_per_token,
                int(datetime.now().timestamp()),
            ]
        )


def print_status(
    generation: int,
    step: int,
    loss_nats_per_token: float,
    tries: int,
) -> None:
    loss_bits_per_token = loss_nats_per_token / math.log(2.0)

    print(
        "Generation "
        f"{generation}. "
        f"(Steps: {step}). "
        f"Loss: {loss_bits_per_token:.6f} bits/token. "
        f"Discarded siblings: {tries - 1}",
        flush=True,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", type=str, default="./training.txt")
    parser.add_argument("--vocab-size", type=int, default=MODEL_VOCAB_SIZE)
    parser.add_argument("--num-ff", type=int, default=DEFAULT_NUM_FF)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--checkpoint-path", type=str, default="./checkpoint.pt")
    parser.add_argument("--csv-path", type=str, default="./data.csv")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    if args.vocab_size < 1 or args.vocab_size > MODEL_VOCAB_SIZE:
        raise ValueError(f"--vocab-size must be in [1, {MODEL_VOCAB_SIZE}]")

    if args.num_ff < 0:
        raise ValueError("--num-ff must be >= 0")

    if args.print_every < 0:
        raise ValueError("--print-every must be >= 0")

    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be >= 0")

    # Configure device and seeds.
    device = resolve_device(args.device)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"device: {device}")

    # Load training data.
    print(f"Training data: {args.file}")
    encoded_cpu, byte_to_id, id_to_byte = load_byte_training_file(
        args.file,
        args.vocab_size,
    )

    batch = make_training_batch(encoded_cpu, device)

    # Load checkpoint or initialize new model.
    if args.resume:
        model, step, generation, ckpt_byte_to_id, ckpt_id_to_byte = load_checkpoint(
            path=args.checkpoint_path,
            device=device,
        )

        if ckpt_byte_to_id and ckpt_byte_to_id != byte_to_id:
            raise ValueError(
                "Checkpoint byte_to_id mapping does not match the current training file. "
                "Use the same data/vocabulary that produced the checkpoint."
            )

        if ckpt_id_to_byte and ckpt_id_to_byte != id_to_byte:
            raise ValueError(
                "Checkpoint id_to_byte mapping does not match the current training file. "
                "Use the same data/vocabulary that produced the checkpoint."
            )
    else:
        model = KernelBRNN(num_ff=args.num_ff)
        model.eval()
        step = 0
        generation = 0

    print(
        f"Batch shape: {tuple(batch.shape)}; "
        f"vocab size: {len(byte_to_id)}; "
        f"num_ff: {model.num_ff}; "
        f"starting step: {step}; "
        f"starting generation: {generation}"
    )

    # Commence genetic algorithm.
    while True:
        prev_step = step

        loss_nats_per_token, tries = genetic_alg_step(model, batch)

        step += tries
        generation += 1

        append_csv_row(
            csv_path=args.csv_path,
            generation=generation,
            step=step,
            loss_nats_per_token=loss_nats_per_token,
        )

        if args.print_every > 0 and (
            generation == 1 or generation % args.print_every == 0
        ):
            print_status(
                generation=generation,
                step=step,
                loss_nats_per_token=loss_nats_per_token,
                tries=tries,
            )

        if args.checkpoint_every > 0 and (
            step // args.checkpoint_every > prev_step // args.checkpoint_every
        ):
            save_checkpoint(
                path=args.checkpoint_path,
                model=model,
                byte_to_id=byte_to_id,
                id_to_byte=id_to_byte,
                step=step,
                generation=generation,
            )


if __name__ == "__main__":
    main()
