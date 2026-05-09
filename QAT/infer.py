import argparse
from typing import Dict

import torch

from model import BRNN, DEFAULT_NUM_FF, infer_carry_dim_from_state_dict


def torch_load_checkpoint(path: str):
    """
    Compatibility wrapper.

    Newer PyTorch versions support weights_only. Older versions do not.
    This checkpoint contains Python RNG state, so weights_only=False is
    needed when the argument exists.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_mapping_keys(d: Dict) -> Dict[int, int]:
    return {int(k): int(v) for k, v in d.items()}


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch_load_checkpoint(checkpoint_path)

    if "latent_state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint at {checkpoint_path} does not contain "
            f"'latent_state_dict' (was it produced by train.py?)."
        )

    num_ff = int(checkpoint.get("num_ff", DEFAULT_NUM_FF))

    if "carry_dim" in checkpoint:
        carry_dim = int(checkpoint["carry_dim"])
    else:
        carry_dim = infer_carry_dim_from_state_dict(checkpoint["latent_state_dict"])

    model = BRNN(num_ff=num_ff, carry_dim=carry_dim).to(device)

    state_dict = {
        k: v.to(device) for k, v in checkpoint["latent_state_dict"].items()
    }
    model.load_state_dict(state_dict)
    model.eval()

    byte_to_id = normalize_mapping_keys(checkpoint.get("byte_to_id", {}))
    id_to_byte = normalize_mapping_keys(checkpoint.get("id_to_byte", {}))

    return model, byte_to_id, id_to_byte


def encode_prompt(
    prompt: str,
    byte_to_id: Dict[int, int],
    device: torch.device,
) -> torch.Tensor:
    prompt_bytes = prompt.encode("utf-8")

    token_ids = []
    for b in prompt_bytes:
        if b not in byte_to_id:
            raise ValueError(
                f"Prompt contains byte {b}, which is not present in the "
                f"checkpoint vocabulary."
            )
        token_ids.append(byte_to_id[b])

    return torch.tensor(token_ids, dtype=torch.long, device=device)


def decode_tokens(tokens, id_to_byte: Dict[int, int]) -> str:
    raw = bytearray()

    for token in tokens:
        token = int(token)

        if token not in id_to_byte:
            raw.extend(b"?")
            continue

        raw.append(int(id_to_byte[token]))

    return bytes(raw).decode("utf-8", errors="replace")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint-path", type=str, default="./checkpoint.pt")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="0 = greedy argmax; >0 = sample at this temperature")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    if args.num_tokens < 0:
        raise ValueError("--num-tokens must be >= 0")
    if args.temperature < 0:
        raise ValueError("--temperature must be >= 0")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    model, byte_to_id, id_to_byte = load_model(args.checkpoint_path, device)

    prompt_tokens = encode_prompt(args.prompt, byte_to_id, device)

    generated = model.generate(
        prompt_tokens=prompt_tokens,
        num_tokens=args.num_tokens,
        temperature=args.temperature,
    )

    generated_text = decode_tokens(generated.tolist(), id_to_byte)

    print(args.prompt + generated_text)


if __name__ == "__main__":
    main()
