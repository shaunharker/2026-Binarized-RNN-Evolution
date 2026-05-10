import argparse
from typing import Dict, Tuple

import torch

from model import ReferenceBRNN, DEFAULT_NUM_FF


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


def normalize_mapping_keys(d: Dict) -> Dict[int, int]:
    return {int(k): int(v) for k, v in d.items()}


def load_reference_model(
    checkpoint_path: str,
) -> Tuple[ReferenceBRNN, Dict[int, int], Dict[int, int]]:
    checkpoint = torch_load_checkpoint(checkpoint_path)
    state_dict = checkpoint["model_state_dict"]

    if "num_ff" in checkpoint:
        num_ff = int(checkpoint["num_ff"])
    elif "ff" in state_dict:
        num_ff = int(state_dict["ff"].shape[0])
    else:
        num_ff = DEFAULT_NUM_FF

    model = ReferenceBRNN(num_ff=num_ff)
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
                f"Prompt contains byte {b}, which is not present in the checkpoint vocabulary."
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
def generate_tokens(
    model: ReferenceBRNN,
    prompt_tokens: torch.Tensor,
    num_tokens: int,
    temperature: float,
) -> list[int]:
    x = model.initial.clone()

    for token in prompt_tokens:
        x = model.advance(x, token)

    generated = []

    for _ in range(num_tokens):
        token, x = model.generate(
            prompt=None,
            activation=x,
            temperature=temperature,
        )

        generated.append(token)

    return generated


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint-path", type=str, default="./checkpoint.pt")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    if args.num_tokens < 0:
        raise ValueError("--num-tokens must be >= 0")

    if args.temperature < 0:
        raise ValueError("--temperature must be >= 0")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, byte_to_id, id_to_byte = load_reference_model(args.checkpoint_path)

    device = model.initial.device
    prompt_tokens = encode_prompt(args.prompt, byte_to_id, device)

    generated_tokens = generate_tokens(
        model=model,
        prompt_tokens=prompt_tokens,
        num_tokens=args.num_tokens,
        temperature=args.temperature,
    )

    generated_text = decode_tokens(generated_tokens, id_to_byte)

    # print(f"checkpoint: {args.checkpoint_path}")
    # print(f"num_ff: {model.num_ff}")
    # print(f"prompt: {args.prompt!r}")
    # print(f"num_tokens: {args.num_tokens}")
    # print(f"temperature: {args.temperature}")
    # print(f"generated_tokens: {generated_tokens}")
    # print()
    print(args.prompt + generated_text)


if __name__ == "__main__":
    main()
