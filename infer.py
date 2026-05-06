## `infer.py`

```python
import argparse
from typing import Dict

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


def load_reference_model(checkpoint_path: str) -> tuple[ReferenceBRNN, Dict[int, int], Dict[int, int]]:
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


def encode_prompt(prompt: str, byte_to_id: Dict[int, int], device: torch.device) -> torch.Tensor:
    prompt_bytes = prompt.encode("utf-8")

    token_ids = []

    for b in prompt_bytes:
        if b not in byte_to_id:
            raise ValueError(
                f"Prompt contains byte {b!r}, which is not present in the checkpoint vocabulary."
            )

        token_ids.append(byte_to_id[b])

    return torch.tensor(token_ids, dtype=torch.long, device=device)


def format_predicted_byte(token: int, id_to_byte: Dict[int, int]) -> str:
    if token not in id_to_byte:
        return f"<token {token}; no byte mapping>"

    b = id_to_byte[token]
    raw = bytes([b])

    try:
        text = raw.decode("utf-8")
        return f"{text!r} byte={b} token={token}"
    except UnicodeDecodeError:
        return f"{raw!r} byte={b} token={token}"


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint-path", type=str, default="./checkpoint.pt")
    parser.add_argument("--prompt", type=str, default="")

    args = parser.parse_args()

    model, byte_to_id, id_to_byte = load_reference_model(args.checkpoint_path)

    device = model.initial.device
    prompt_tokens = encode_prompt(args.prompt, byte_to_id, device)

    token, _ = model.generate(prompt=prompt_tokens)

    print(f"checkpoint: {args.checkpoint_path}")
    print(f"num_ff: {model.num_ff}")
    print(f"prompt: {args.prompt!r}")
    print(f"prompt_tokens: {prompt_tokens.detach().cpu().tolist()}")
    print(f"predicted_next: {format_predicted_byte(token, id_to_byte)}")


if __name__ == "__main__":
    main()
