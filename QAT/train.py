# train.py

import argparse
import csv
import math
import os
import random
import time
from datetime import datetime
from typing import Dict, Tuple

import torch

from model import (
    BRNN,
    DEFAULT_NUM_FF,
    DEFAULT_CARRY_DIM,
    VOCAB,
    infer_carry_dim_from_state_dict,
)


MODEL_VOCAB_SIZE = VOCAB  # 128


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

    distinct = list(range(128))  # actually, let's just do ascii.

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


def make_random_batch(
    encoded_cpu: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Pick `batch_size` random contiguous snippets of length `seq_len`."""
    n = encoded_cpu.numel()
    if n <= seq_len:
        raise ValueError(
            f"Training file has only {n} encoded tokens, "
            f"but we need more than seq_len={seq_len} to sample snippets."
        )

    starts = torch.randint(0, n - seq_len, (batch_size,))
    offsets = torch.arange(seq_len)
    indices = starts.unsqueeze(1) + offsets.unsqueeze(0)  # [B, T]

    return encoded_cpu[indices].to(device=device, dtype=torch.long, non_blocking=True)


# ============================================================
# Device
# ============================================================

def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_arg)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        if device.index is not None:
            torch.cuda.set_device(device)
        device = torch.device("cuda", torch.cuda.current_device())

    return device


# ============================================================
# Checkpoint Saving and Loading
# ============================================================

def save_checkpoint(
    path: str,
    model: BRNN,
    optimizer: torch.optim.Optimizer,
    byte_to_id: Dict[int, int],
    id_to_byte: Dict[int, int],
    step: int,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    latent_state_cpu = {
        k: v.detach().cpu() for k, v in model.state_dict().items()
    }

    quantized = model.export_quantized()
    cuda_rng = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    )

    payload = {
        "step":      int(step),
        "num_ff":    int(model.num_ff),
        "carry_dim": int(model.carry_dim),
        "read_dim":  int(model.read_dim),
        "latent_state_dict":    latent_state_cpu,
        "quantized":            quantized,
        "optimizer_state_dict": optimizer.state_dict(),
        "byte_to_id": byte_to_id,
        "id_to_byte": id_to_byte,
        "torch_rng_state":     torch.get_rng_state(),
        "cuda_rng_state_all":  cuda_rng,
        "python_random_state": random.getstate(),
    }

    torch.save(payload, path)

    now = datetime.now()
    print(f"\n>>> Checkpoint saved to {path} at step {step}")
    print(f">>> Time: {now.strftime('%Y-%m-%d %H:%M:%S')} ({int(now.timestamp())})")


def torch_load_checkpoint(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_checkpoint(
    path: str,
    device: torch.device,
):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    print(f"Loading checkpoint: {path}")
    checkpoint = torch_load_checkpoint(path)

    num_ff = int(checkpoint.get("num_ff", DEFAULT_NUM_FF))

    if "carry_dim" in checkpoint:
        carry_dim = int(checkpoint["carry_dim"])
    else:
        carry_dim = infer_carry_dim_from_state_dict(checkpoint["latent_state_dict"])

    model = BRNN(num_ff=num_ff, carry_dim=carry_dim).to(device)
    model.load_state_dict(
        {k: v.to(device) for k, v in checkpoint["latent_state_dict"].items()}
    )

    step = int(checkpoint.get("step", 0))
    byte_to_id = checkpoint.get("byte_to_id", {})
    id_to_byte = checkpoint.get("id_to_byte", {})
    optim_state = checkpoint.get("optimizer_state_dict", None)

    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])

    if "cuda_rng_state_all" in checkpoint and torch.cuda.is_available():
        cuda_states = checkpoint["cuda_rng_state_all"]
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)

    if "python_random_state" in checkpoint:
        random.setstate(checkpoint["python_random_state"])

    print(
        f"Resumed from step {step}, num_ff {model.num_ff}, "
        f"carry_dim {model.carry_dim}, d_model {model.d_model}."
    )
    return model, step, byte_to_id, id_to_byte, optim_state


# ============================================================
# Logging
# ============================================================

def append_csv_row(
    csv_path: str,
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
                    "step",
                    "loss_nats_per_token",
                    "loss_bits_per_token",
                    "unixtime",
                ]
            )

        writer.writerow(
            [
                step,
                loss_nats_per_token,
                loss_bits_per_token,
                int(datetime.now().timestamp()),
            ]
        )


def print_status(
    step: int,
    loss_nats_per_token: float,
    steps_per_sec: float,
) -> None:
    loss_bits_per_token = loss_nats_per_token / math.log(2.0)
    print(
        f"Step {step}. "
        f"Loss: {loss_bits_per_token:.6f} bits/token. "
        f"({steps_per_sec:.1f} step/s)",
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
    parser.add_argument("--carry-dim", type=int, default=DEFAULT_CARRY_DIM,
                        help="carry dimension; d_model = carry_dim + 128")

    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seq-len", type=int, default=256)

    parser.add_argument("--lr", type=float, default=1e-4,
                        help="LR for sign-quantised weights")
    parser.add_argument("--thresh-lr", type=float, default=1e-3,
                        help="LR for integer-quantised thresholds")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--bf16", action="store_true",
                        help="train under bfloat16 autocast (CUDA only)")
    parser.add_argument("--no-tf32", action="store_true",
                        help="disable TF32 matmul (TF32 is on by default on Ampere+)")

    parser.add_argument("--steps", type=int, default=0,
                        help="0 = train forever")

    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--csv-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
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
    if args.carry_dim < 0:
        raise ValueError("--carry-dim must be >= 0")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.seq_len < 2:
        raise ValueError("--seq-len must be >= 2")

    # Configure device and seeds.
    device = resolve_device(args.device)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # TF32 + bf16 setup.
    if not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.bf16 and device.type != "cuda":
        raise RuntimeError("--bf16 requires CUDA")
    amp_enabled = bool(args.bf16)

    print(f"device: {device}")

    # Load training data.
    print(f"Training data: {args.file}")
    encoded_cpu, byte_to_id, id_to_byte = load_byte_training_file(
        args.file,
        args.vocab_size,
    )
    print(
        f"Loaded {encoded_cpu.numel()} encoded tokens; "
        f"{len(byte_to_id)} distinct symbols."
    )

    # Build / resume model.
    if args.resume:
        model, step, ckpt_byte_to_id, ckpt_id_to_byte, optim_state = load_checkpoint(
            path=args.checkpoint_path,
            device=device,
        )

        if args.carry_dim != DEFAULT_CARRY_DIM and args.carry_dim != model.carry_dim:
            print(
                f"warning: --carry-dim {args.carry_dim} ignored; "
                f"using {model.carry_dim} from checkpoint."
            )
        if args.num_ff != DEFAULT_NUM_FF and args.num_ff != model.num_ff:
            print(
                f"warning: --num-ff {args.num_ff} ignored; "
                f"using {model.num_ff} from checkpoint."
            )

        if ckpt_byte_to_id and ckpt_byte_to_id != byte_to_id:
            raise ValueError(
                "Checkpoint byte_to_id mapping does not match the current training file."
            )
        if ckpt_id_to_byte and ckpt_id_to_byte != id_to_byte:
            raise ValueError(
                "Checkpoint id_to_byte mapping does not match the current training file."
            )
    else:
        model = BRNN(num_ff=args.num_ff, carry_dim=args.carry_dim).to(device)
        step = 0
        optim_state = None

    # Two param groups: sign-latents and threshold-latents (different LRs).
    sign_params = [
        model.initial_lat,
        model.embed_lat,
        model.ff_lat,
        model.head_lat,
    ]
    thresh_params = [model.ff_thresh_lat]

    optimizer = torch.optim.AdamW(
        [
            {"params": sign_params,   "lr": args.lr,        "weight_decay": args.weight_decay},
            {"params": thresh_params, "lr": args.thresh_lr, "weight_decay": 0.0},
        ]
    )

    if optim_state is not None:
        optimizer.load_state_dict(optim_state)

    print(
        f"Batch shape: ({args.batch_size}, {args.seq_len}); "
        f"vocab size: {len(byte_to_id)}; "
        f"num_ff: {model.num_ff}; "
        f"carry_dim: {model.carry_dim}; "
        f"d_model: {model.d_model}; "
        f"bf16: {amp_enabled}; "
        f"starting step: {step}"
    )

    # Train.
    model.train()

    t0 = time.time()
    log_t0 = t0
    log_step0 = step

    while True:
        if args.steps > 0 and step >= args.steps:
            break

        batch = make_random_batch(
            encoded_cpu,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            device=device,
        )

        if amp_enabled:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(batch)
        else:
            loss = model(batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        model.clip_latents_()

        step += 1
        loss_val = float(loss.item())

        if args.csv_every > 0 and step % args.csv_every == 0:
            append_csv_row(
                csv_path=args.csv_path,
                step=step,
                loss_nats_per_token=loss_val,
            )

        if args.print_every > 0 and step % args.print_every == 0:
            now = time.time()
            sps = (step - log_step0) / max(now - log_t0, 1e-6)
            print_status(
                step=step,
                loss_nats_per_token=loss_val,
                steps_per_sec=sps,
            )
            log_t0 = now
            log_step0 = step

        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            save_checkpoint(
                path=args.checkpoint_path,
                model=model,
                optimizer=optimizer,
                byte_to_id=byte_to_id,
                id_to_byte=id_to_byte,
                step=step,
            )

    # Final save.
    save_checkpoint(
        path=args.checkpoint_path,
        model=model,
        optimizer=optimizer,
        byte_to_id=byte_to_id,
        id_to_byte=id_to_byte,
        step=step,
    )


if __name__ == "__main__":
    main()
