import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load


_EXT = None

# ============================================================================
# Architecture constants
# ============================================================================

CARRY_DIM = 384
READ_DIM = 128
D_MODEL = CARRY_DIM + READ_DIM   # 512
VOCAB = 128

D_MODEL_WORDS = D_MODEL // 32    # 16
CARRY_WORDS = CARRY_DIM // 32    # 12
READ_WORDS = READ_DIM // 32      # 4

DEFAULT_NUM_FF = 7

BASE_SHAPES = {
    "initial": (D_MODEL,),
    "embed": (VOCAB, READ_DIM),
    "head": (READ_DIM, VOCAB),
}


# =============================================================================
# CUDA extension
# =============================================================================

def get_extension(verbose=False):
    global _EXT

    if _EXT is not None:
        return _EXT

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for KernelBRNN")

    dev = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(dev)
    arch = f"{major}{minor}"

    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brnn_cuda.cu")

    _EXT = load(
        name=f"brnn_cuda_ext_v4t_sm{arch}",
        sources=[src],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            f"--generate-code=arch=compute_{arch},code=sm_{arch}",
        ],
        verbose=verbose,
    )

    return _EXT


# =============================================================================
# Helpers
# =============================================================================

def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def random_pm1(shape, device):
    return torch.randint(0, 2, shape, device=device, dtype=torch.int8).mul_(2).sub_(1)


def require_shape(x, shape, name):
    if tuple(x.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}")


def require_pm1(x, name):
    if not torch.logical_or(x == 1, x == -1).all().item():
        raise ValueError(f"{name} must contain only -1 and +1")


def has_attrs(obj, names):
    return all(hasattr(obj, name) for name in names)


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def infer_num_ff(source, fallback):
    if source is None:
        return fallback

    if hasattr(source, "num_ff"):
        return int(source.num_ff)

    if hasattr(source, "ff"):
        return int(source.ff.shape[0])

    if hasattr(source, "ff_p"):
        return int(source.ff_p.shape[0])

    return fallback


def shapes_for_num_ff(num_ff):
    return {
        "initial": (D_MODEL,),
        "embed": (VOCAB, READ_DIM),
        "ff": (num_ff, D_MODEL, D_MODEL),
        "head": (READ_DIM, VOCAB),
        "ff_thresh": (num_ff, D_MODEL),
    }


def normalize_index(shape, index):
    if isinstance(index, torch.Tensor):
        index = index.detach().cpu().tolist()

    if isinstance(index, int):
        index = (index,)
    else:
        index = tuple(int(i) for i in index)

    if len(index) != len(shape):
        raise ValueError(f"index {index} does not match shape {shape}")

    for i, s in zip(index, shape):
        if i < 0 or i >= s:
            raise IndexError(f"index {index} out of bounds for shape {shape}")

    return index


def parse_ff_target(target, num_ff):
    if target == "ff":
        return None

    if isinstance(target, str) and target.startswith("ff") and target[2:].isdigit():
        layer = int(target[2:])

        if layer < 0 or layer >= num_ff:
            raise IndexError(f"{target} out of bounds for num_ff={num_ff}")

        return layer

    return None


def pick_mutation(num_ff, target=None, index=None):
    """
    Pick a mutation site (target name + index).

    Note: this does NOT pick a delta for threshold mutations; the calling
    `mutate` method is responsible for picking a +1 / -1 delta.
    """
    shapes = shapes_for_num_ff(num_ff)

    if target is not None:
        ff_layer = parse_ff_target(target, num_ff)

        if target in BASE_SHAPES:
            shape = BASE_SHAPES[target]
            if index is None:
                flat = int(torch.randint(0, numel(shape), ()).item())
                index = np.unravel_index(flat, shape)
            return target, normalize_index(shape, index)

        if target == "ff":
            shape = shapes["ff"]
            if index is None:
                flat = int(torch.randint(0, numel(shape), ()).item())
                index = np.unravel_index(flat, shape)
            return "ff", normalize_index(shape, index)

        if target == "ff_thresh":
            shape = shapes["ff_thresh"]
            if index is None:
                flat = int(torch.randint(0, numel(shape), ()).item())
                index = np.unravel_index(flat, shape)
            return "ff_thresh", normalize_index(shape, index)

        if ff_layer is not None:
            shape = (D_MODEL, D_MODEL)
            if index is None:
                flat = int(torch.randint(0, numel(shape), ()).item())
                index = np.unravel_index(flat, shape)
            i, j = normalize_index(shape, index)
            return "ff", (ff_layer, i, j)

        raise ValueError(
            "target must be one of "
            "['initial', 'embed', 'ff', 'head', 'ff_thresh'] "
            "or a layer name like 'ff0', 'ff1', ..."
        )

    if index is not None:
        raise ValueError("index requires target")

    total = sum(numel(s) for s in shapes.values())
    flat = int(torch.randint(0, total, ()).item())

    for name, shape in shapes.items():
        n = numel(shape)
        if flat < n:
            return name, normalize_index(shape, np.unravel_index(flat, shape))
        flat -= n

    raise RuntimeError("unreachable")


def random_pm1_delta():
    """Pick a random delta in {-1, +1}."""
    return 1 if int(torch.randint(0, 2, ()).item()) == 0 else -1


def int32_bit_mask(bit):
    return np.array(1 << bit, dtype=np.uint32).view(np.int32).item()


# =============================================================================
# Packing
# =============================================================================

def pack_pm1_rows(mat):
    """
    Pack int8 -1/+1 rows into int32 words.

    Convention:
        +1 -> bit 1
        -1 -> bit 0
    """
    if mat.ndim != 2:
        raise ValueError("mat must be rank 2")

    rows, cols = mat.shape
    if cols % 32:
        raise ValueError("number of columns must be divisible by 32")

    require_pm1(mat, "mat")

    bits = (mat.detach().cpu().numpy() > 0).astype(np.uint8)
    packed = np.packbits(bits, axis=1, bitorder="little")
    packed = np.ascontiguousarray(packed).view("<u4").reshape(rows, cols // 32)

    return torch.from_numpy(packed.view(np.int32)).to(device=mat.device)


def unpack_pm1_rows(packed, cols):
    if packed.ndim != 2:
        raise ValueError("packed must be rank 2")
    if cols % 32:
        raise ValueError("cols must be divisible by 32")

    rows, words = packed.shape
    if words != cols // 32:
        raise ValueError(f"expected {cols // 32} words, got {words}")

    arr = packed.detach().cpu().contiguous().numpy()
    arr = arr.view(np.uint32).view(np.uint8).reshape(rows, words * 4)

    bits = np.unpackbits(arr, axis=1, bitorder="little")[:, :cols]
    pm1 = bits.astype(np.int8) * 2 - 1

    return torch.from_numpy(pm1).to(device=packed.device)


def pack_initial(x):
    require_shape(x, (D_MODEL,), "initial")
    return pack_pm1_rows(x.view(1, D_MODEL)).view(D_MODEL_WORDS)


def unpack_initial(x):
    require_shape(x, (D_MODEL_WORDS,), "initial_p")
    return unpack_pm1_rows(x.view(1, D_MODEL_WORDS), D_MODEL).view(D_MODEL)


def pack_embed(x):
    require_shape(x, (VOCAB, READ_DIM), "embed")
    return pack_pm1_rows(x)


def unpack_embed(x):
    require_shape(x, (VOCAB, READ_WORDS), "embed_p")
    return unpack_pm1_rows(x, READ_DIM)


def pack_linear_weight_columns_word_major(w):
    """
    Logical weight: [in_dim, out_dim]
    Packed layout: [in_dim // 32, out_dim]
    """
    if w.ndim != 2:
        raise ValueError("weight must be rank 2")

    return pack_pm1_rows(w.t().contiguous()).t().contiguous()


def unpack_linear_weight_columns_word_major(packed, in_dim, out_dim):
    require_shape(packed, (in_dim // 32, out_dim), "packed")
    return unpack_pm1_rows(packed.t().contiguous(), in_dim).t().contiguous()


def pack_ff(x):
    if x.ndim != 3:
        raise ValueError(f"ff must have shape [num_ff, {D_MODEL}, {D_MODEL}]")

    if x.shape[1:] != (D_MODEL, D_MODEL):
        raise ValueError(f"ff must have shape [num_ff, {D_MODEL}, {D_MODEL}]")

    if x.shape[0] == 0:
        return torch.empty((0, D_MODEL_WORDS, D_MODEL), device=x.device, dtype=torch.int32)

    return torch.stack(
        [pack_linear_weight_columns_word_major(x[i]) for i in range(x.shape[0])],
        dim=0,
    ).contiguous()


def unpack_ff(x):
    if x.ndim != 3:
        raise ValueError(f"ff_p must have shape [num_ff, {D_MODEL_WORDS}, {D_MODEL}]")

    if x.shape[1:] != (D_MODEL_WORDS, D_MODEL):
        raise ValueError(f"ff_p must have shape [num_ff, {D_MODEL_WORDS}, {D_MODEL}]")

    if x.shape[0] == 0:
        return torch.empty((0, D_MODEL, D_MODEL), device=x.device, dtype=torch.int8)

    return torch.stack(
        [
            unpack_linear_weight_columns_word_major(
                x[i],
                in_dim=D_MODEL,
                out_dim=D_MODEL,
            )
            for i in range(x.shape[0])
        ],
        dim=0,
    ).contiguous()


# =============================================================================
# Reference PyTorch model
# =============================================================================

class ReferenceBRNN(nn.Module):
    """
    Unpacked PyTorch implementation.

    Buffers:
        initial:    [512]                int8, +-1
        embed:      [128, 128]           int8, +-1
        ff:         [num_ff, 512, 512]   int8, +-1
        head:       [128, 128]           int8, +-1
        ff_thresh:  [num_ff, 512]        int32, integer thresholds
    """

    def __init__(self, source=None, num_ff=None):
        super().__init__()

        device = default_device()

        if num_ff is None:
            num_ff = infer_num_ff(source, DEFAULT_NUM_FF)

        self.num_ff = int(num_ff)

        if self.num_ff < 0:
            raise ValueError("num_ff must be >= 0")

        self.register_buffer(
            "initial", torch.empty((D_MODEL,), device=device, dtype=torch.int8)
        )
        self.register_buffer(
            "embed", torch.empty((VOCAB, READ_DIM), device=device, dtype=torch.int8)
        )
        self.register_buffer(
            "ff",
            torch.empty((self.num_ff, D_MODEL, D_MODEL), device=device, dtype=torch.int8),
        )
        self.register_buffer(
            "head", torch.empty((READ_DIM, VOCAB), device=device, dtype=torch.int8)
        )
        self.register_buffer(
            "ff_thresh",
            torch.zeros((self.num_ff, D_MODEL), device=device, dtype=torch.int32),
        )

        if source is None:
            self.initial.copy_(random_pm1((D_MODEL,), device))
            self.embed.copy_(random_pm1((VOCAB, READ_DIM), device))
            self.ff.copy_(random_pm1((self.num_ff, D_MODEL, D_MODEL), device))
            self.head.copy_(random_pm1((READ_DIM, VOCAB), device))
            self.ff_thresh.zero_()
        else:
            self._copy_from(source)

    @torch.no_grad()
    def _copy_from(self, source):
        if has_attrs(source, ["initial", "embed", "ff", "head"]):
            require_shape(source.initial, (D_MODEL,), "source.initial")
            require_shape(source.embed, (VOCAB, READ_DIM), "source.embed")
            require_shape(source.ff, (self.num_ff, D_MODEL, D_MODEL), "source.ff")
            require_shape(source.head, (READ_DIM, VOCAB), "source.head")

            self.initial.copy_(source.initial.to(device=self.initial.device, dtype=torch.int8))
            self.embed.copy_(source.embed.to(device=self.embed.device, dtype=torch.int8))
            self.ff.copy_(source.ff.to(device=self.ff.device, dtype=torch.int8))
            self.head.copy_(source.head.to(device=self.head.device, dtype=torch.int8))

            if hasattr(source, "ff_thresh"):
                require_shape(
                    source.ff_thresh, (self.num_ff, D_MODEL), "source.ff_thresh"
                )
                self.ff_thresh.copy_(
                    source.ff_thresh.to(device=self.ff_thresh.device, dtype=torch.int32)
                )
            else:
                self.ff_thresh.zero_()

            return

        if has_attrs(source, ["initial_p", "embed_p", "ff_p", "head_p"]):
            require_shape(source.initial_p, (D_MODEL_WORDS,), "source.initial_p")
            require_shape(source.embed_p, (VOCAB, READ_WORDS), "source.embed_p")
            require_shape(
                source.ff_p, (self.num_ff, D_MODEL_WORDS, D_MODEL), "source.ff_p"
            )
            require_shape(source.head_p, (READ_WORDS, VOCAB), "source.head_p")

            self.initial.copy_(
                unpack_initial(source.initial_p).to(device=self.initial.device, dtype=torch.int8)
            )

            self.embed.copy_(
                unpack_embed(source.embed_p).to(device=self.embed.device, dtype=torch.int8)
            )

            self.ff.copy_(
                unpack_ff(source.ff_p).to(device=self.ff.device, dtype=torch.int8)
            )

            self.head.copy_(
                unpack_linear_weight_columns_word_major(
                    source.head_p,
                    in_dim=READ_DIM,
                    out_dim=VOCAB,
                ).to(device=self.head.device, dtype=torch.int8)
            )

            if hasattr(source, "ff_thresh"):
                require_shape(
                    source.ff_thresh, (self.num_ff, D_MODEL), "source.ff_thresh"
                )
                self.ff_thresh.copy_(
                    source.ff_thresh.to(device=self.ff_thresh.device, dtype=torch.int32)
                )
            else:
                self.ff_thresh.zero_()

            return

        raise TypeError(
            "source must be ReferenceBRNN-like with buffers "
            "`initial`, `embed`, `ff`, `head` (and optionally `ff_thresh`), "
            "or KernelBRNN-like with buffers "
            "`initial_p`, `embed_p`, `ff_p`, `head_p` (and optionally `ff_thresh`)"
        )

    @staticmethod
    def _int_mm(a, b):
        a = a.contiguous()
        b = b.contiguous()

        if (
            hasattr(torch, "_int_mm")
            and a.is_cuda
            and b.is_cuda
            and a.dtype == torch.int8
            and b.dtype == torch.int8
            and a.shape[0] > 16
        ):
            return torch._int_mm(a, b)

        if not a.is_cuda:
            return a.to(torch.int32) @ b.to(torch.int32)

        return (
            a.to(torch.int32).unsqueeze(2)
            * b.to(torch.int32).unsqueeze(0)
        ).sum(dim=1)

    @staticmethod
    def _sign_with_thresh(preact, thresh):
        """
        preact: [B, D_MODEL] int32
        thresh: [D_MODEL]    int32

        Returns int8 +/-1.
        """
        return torch.where(
            preact > thresh,
            torch.ones((), dtype=torch.int8, device=preact.device),
            -torch.ones((), dtype=torch.int8, device=preact.device),
        ).contiguous()

    @torch.no_grad()
    def logits_and_carry(self, x):
        single = False

        if x.ndim == 1:
            require_shape(x, (D_MODEL,), "x")
            x = x.view(1, D_MODEL)
            single = True

        if x.ndim != 2 or x.shape[1] != D_MODEL:
            raise ValueError(f"x must have shape [{D_MODEL}] or [B, {D_MODEL}]")

        x = x.to(device=self.initial.device, dtype=torch.int8).contiguous()

        for layer in range(self.num_ff):
            preact = self._int_mm(x, self.ff[layer])  # [B, D_MODEL] int32
            x = self._sign_with_thresh(preact, self.ff_thresh[layer])

        carry = x[:, :CARRY_DIM].contiguous()
        read = x[:, CARRY_DIM:].contiguous()
        logits = self._int_mm(read, self.head).to(torch.float32).mul_(1.0 / 16.0)

        return (logits[0], carry[0]) if single else (logits, carry)

    @torch.no_grad()
    def advance(self, x, token):
        _, carry = self.logits_and_carry(x)

        if x.ndim == 1:
            token = int(token.item()) if isinstance(token, torch.Tensor) else int(token)
            if token < 0 or token >= VOCAB:
                raise ValueError(f"token must be in [0, {VOCAB - 1}]")
            return torch.cat([carry, self.embed[token]], dim=0).contiguous()

        tokens = torch.as_tensor(
            token,
            dtype=torch.long,
            device=self.initial.device,
        ).flatten()

        if tokens.shape != (x.shape[0],):
            raise ValueError(f"token must have shape [{x.shape[0]}]")

        return torch.cat([carry, self.embed[tokens]], dim=1).contiguous()

    @torch.no_grad()
    def loss(self, tokens):
        tokens = torch.as_tensor(tokens, dtype=torch.long, device=self.initial.device)

        single = False
        if tokens.ndim == 1:
            tokens = tokens.view(1, -1)
            single = True

        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [B, T] or [T]")

        B, T = tokens.shape

        x = self.initial.unsqueeze(0).expand(B, D_MODEL).clone().contiguous()
        losses = torch.zeros(B, dtype=torch.float32, device=self.initial.device)

        for t in range(T):
            logits, carry = self.logits_and_carry(x)
            losses += F.cross_entropy(logits, tokens[:, t], reduction="none")
            x = torch.cat([carry, self.embed[tokens[:, t]]], dim=1).contiguous()

        return losses[0] if single else losses

    forward = loss

    @torch.no_grad()
    def generate(self, prompt=None, activation=None, temperature=1.0):
        device = self.initial.device

        if temperature < 0:
            raise ValueError("temperature must be >= 0")

        if prompt is None:
            prompt = torch.empty(0, dtype=torch.long, device=device)
        else:
            prompt = torch.as_tensor(prompt, dtype=torch.long, device=device).flatten()

        if activation is None:
            x = self.initial.clone()
        else:
            require_shape(activation, (D_MODEL,), "activation")
            x = activation.to(device=device, dtype=torch.int8).contiguous().clone()

        for tok in prompt:
            x = self.advance(x, tok)

        logits, _ = self.logits_and_carry(x)

        if temperature == 0:
            token = int(torch.argmax(logits).item())
        else:
            probs = torch.softmax(logits / float(temperature), dim=0)
            token = int(torch.multinomial(probs, num_samples=1).item())

        return token, self.advance(x, token)

    @torch.no_grad()
    def _apply_pm1_flip(self, target, index):
        if target == "ff":
            self.ff[index].mul_(-1)
        else:
            getattr(self, target)[index].mul_(-1)

    @torch.no_grad()
    def _apply_thresh_delta(self, index, delta):
        self.ff_thresh[index] = self.ff_thresh[index] + int(delta)

    @torch.no_grad()
    def mutate(self, descriptor=None):
        """
        With no argument: pick and apply a random mutation.
        Returns a 3-tuple (target, index, delta) describing what was applied.
            * For pm1 flips, delta is 0.
            * For threshold mutations, delta is +1 or -1.

        With a descriptor argument: revert that mutation.
        """
        if descriptor is None:
            target, index = pick_mutation(self.num_ff)

            if target == "ff_thresh":
                delta = random_pm1_delta()
                self._apply_thresh_delta(index, delta)
                return (target, index, delta)

            self._apply_pm1_flip(target, index)
            return (target, index, 0)

        target, index, delta = descriptor

        if target == "ff_thresh":
            self._apply_thresh_delta(index, -int(delta))
        else:
            # bit flip is its own inverse.
            self._apply_pm1_flip(target, index)

        return descriptor


# =============================================================================
# Packed CUDA kernel model
# =============================================================================

class KernelBRNN(nn.Module):
    """
    Packed CUDA-kernel-backed implementation.

    Buffers:
        initial_p:  [16]                int32 (packed bits)
        embed_p:    [128, 4]            int32 (packed bits)
        ff_p:       [num_ff, 16, 512]   int32 (packed bits)
        head_p:     [4, 128]            int32 (packed bits)
        ff_thresh:  [num_ff, 512]       int32 integer thresholds (unpacked)
    """

    def __init__(self, source=None, num_ff=None):
        super().__init__()

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for KernelBRNN")

        device = torch.device("cuda")

        if num_ff is None:
            num_ff = infer_num_ff(source, DEFAULT_NUM_FF)

        self.num_ff = int(num_ff)

        if self.num_ff < 0:
            raise ValueError("num_ff must be >= 0")

        self.extension = get_extension()

        self.register_buffer(
            "initial_p", torch.empty((D_MODEL_WORDS,), device=device, dtype=torch.int32)
        )
        self.register_buffer(
            "embed_p", torch.empty((VOCAB, READ_WORDS), device=device, dtype=torch.int32)
        )
        self.register_buffer(
            "ff_p",
            torch.empty(
                (self.num_ff, D_MODEL_WORDS, D_MODEL), device=device, dtype=torch.int32
            ),
        )
        self.register_buffer(
            "head_p", torch.empty((READ_WORDS, VOCAB), device=device, dtype=torch.int32)
        )
        self.register_buffer(
            "ff_thresh",
            torch.zeros((self.num_ff, D_MODEL), device=device, dtype=torch.int32),
        )

        if source is None:
            initial = random_pm1((D_MODEL,), device)
            embed = random_pm1((VOCAB, READ_DIM), device)
            ff = random_pm1((self.num_ff, D_MODEL, D_MODEL), device)
            head = random_pm1((READ_DIM, VOCAB), device)

            self.initial_p.copy_(pack_initial(initial).contiguous())
            self.embed_p.copy_(pack_embed(embed).contiguous())
            self.ff_p.copy_(pack_ff(ff).contiguous())
            self.head_p.copy_(pack_linear_weight_columns_word_major(head).contiguous())
            self.ff_thresh.zero_()
        else:
            self._copy_from(source)

    @torch.no_grad()
    def _copy_from(self, source):
        if has_attrs(source, ["initial_p", "embed_p", "ff_p", "head_p"]):
            require_shape(source.initial_p, (D_MODEL_WORDS,), "source.initial_p")
            require_shape(source.embed_p, (VOCAB, READ_WORDS), "source.embed_p")
            require_shape(
                source.ff_p, (self.num_ff, D_MODEL_WORDS, D_MODEL), "source.ff_p"
            )
            require_shape(source.head_p, (READ_WORDS, VOCAB), "source.head_p")

            self.initial_p.copy_(
                source.initial_p.to(device=self.initial_p.device, dtype=torch.int32)
            )

            self.embed_p.copy_(
                source.embed_p.to(device=self.embed_p.device, dtype=torch.int32)
            )

            self.ff_p.copy_(
                source.ff_p.to(device=self.ff_p.device, dtype=torch.int32)
            )

            self.head_p.copy_(
                source.head_p.to(device=self.head_p.device, dtype=torch.int32)
            )

            if hasattr(source, "ff_thresh"):
                require_shape(
                    source.ff_thresh, (self.num_ff, D_MODEL), "source.ff_thresh"
                )
                self.ff_thresh.copy_(
                    source.ff_thresh.to(device=self.ff_thresh.device, dtype=torch.int32)
                )
            else:
                self.ff_thresh.zero_()

            return

        if has_attrs(source, ["initial", "embed", "ff", "head"]):
            require_shape(source.initial, (D_MODEL,), "source.initial")
            require_shape(source.embed, (VOCAB, READ_DIM), "source.embed")
            require_shape(source.ff, (self.num_ff, D_MODEL, D_MODEL), "source.ff")
            require_shape(source.head, (READ_DIM, VOCAB), "source.head")

            self.initial_p.copy_(
                pack_initial(
                    source.initial.to(dtype=torch.int8)
                ).to(device=self.initial_p.device, dtype=torch.int32).contiguous()
            )

            self.embed_p.copy_(
                pack_embed(
                    source.embed.to(dtype=torch.int8)
                ).to(device=self.embed_p.device, dtype=torch.int32).contiguous()
            )

            self.ff_p.copy_(
                pack_ff(
                    source.ff.to(dtype=torch.int8)
                ).to(device=self.ff_p.device, dtype=torch.int32).contiguous()
            )

            self.head_p.copy_(
                pack_linear_weight_columns_word_major(
                    source.head.to(dtype=torch.int8)
                ).to(device=self.head_p.device, dtype=torch.int32).contiguous()
            )

            if hasattr(source, "ff_thresh"):
                require_shape(
                    source.ff_thresh, (self.num_ff, D_MODEL), "source.ff_thresh"
                )
                self.ff_thresh.copy_(
                    source.ff_thresh.to(device=self.ff_thresh.device, dtype=torch.int32)
                )
            else:
                self.ff_thresh.zero_()

            return

        raise TypeError(
            "source must be KernelBRNN-like with buffers "
            "`initial_p`, `embed_p`, `ff_p`, `head_p` (and optionally `ff_thresh`), "
            "or ReferenceBRNN-like with buffers "
            "`initial`, `embed`, `ff`, `head` (and optionally `ff_thresh`)"
        )

    @torch.no_grad()
    def loss(self, tokens):
        tokens = torch.as_tensor(tokens, dtype=torch.long, device=self.initial_p.device)

        single = False
        if tokens.ndim == 1:
            tokens = tokens.view(1, -1)
            single = True

        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [B, T] or [T]")

        out = self.extension.forward(
            self.initial_p,
            self.embed_p,
            self.ff_p,
            self.ff_thresh,
            self.head_p,
            tokens.contiguous(),
        )

        return out[0] if single else out

    forward = loss

    @staticmethod
    def _packed_mutation(target, index):
        if target == "initial":
            dim = index[0]
            return "initial_p", (dim // 32,), dim % 32

        if target == "embed":
            token, dim = index
            return "embed_p", (token, dim // 32), dim % 32

        if target == "ff":
            layer, input_dim, output_dim = index
            return "ff_p", (layer, input_dim // 32, output_dim), input_dim % 32

        if target == "head":
            input_dim, output_dim = index
            return "head_p", (input_dim // 32, output_dim), input_dim % 32

        raise RuntimeError("unreachable")

    @torch.no_grad()
    def _apply_pm1_flip(self, target, index):
        name, packed_index, bit = self._packed_mutation(target, index)
        tensor = getattr(self, name)

        mask = torch.tensor(
            int32_bit_mask(bit),
            dtype=torch.int32,
            device=tensor.device,
        )

        tensor[packed_index] = torch.bitwise_xor(tensor[packed_index], mask)

    @torch.no_grad()
    def _apply_thresh_delta(self, index, delta):
        self.ff_thresh[index] = self.ff_thresh[index] + int(delta)

    @torch.no_grad()
    def mutate(self, descriptor=None):
        """
        With no argument: pick and apply a random mutation.
        Returns a 3-tuple (target, index, delta).
            * For pm1 flips, delta is 0.
            * For threshold mutations, delta is +1 or -1.

        With a descriptor argument: revert that mutation.
        """
        if descriptor is None:
            target, index = pick_mutation(self.num_ff)

            if target == "ff_thresh":
                delta = random_pm1_delta()
                self._apply_thresh_delta(index, delta)
                return (target, index, delta)

            self._apply_pm1_flip(target, index)
            return (target, index, 0)

        target, index, delta = descriptor

        if target == "ff_thresh":
            self._apply_thresh_delta(index, -int(delta))
        else:
            self._apply_pm1_flip(target, index)

        return descriptor
