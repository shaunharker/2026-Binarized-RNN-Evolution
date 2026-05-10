# convert.py
import math
import numpy as np
import torch

ckpt = torch.load("checkpoint.pt", map_location="cpu", weights_only=False)
q = ckpt["quantized"]

assert q["num_ff"] == 4
assert q["carry_dim"] == 128
assert q["read_dim"] == 128

# ---------- helpers ----------------------------------------------------------

def pack_pm1_bits(arr_pm1):
    """Pack a +1/-1 tensor to bits, MSB first within each byte (+1 -> 1)."""
    bits = (arr_pm1 > 0).astype(np.uint8).flatten()
    return np.packbits(bits, bitorder="big")

# ---------- initial state (256 bits) ----------------------------------------
initial_packed = pack_pm1_bits(q["initial"].numpy())  # 32 B

# ---------- embedding [128 tokens][128 bits] = 2048 B -----------------------
embed_packed = pack_pm1_bits(q["embed"].numpy())

# ---------- FF weights ------------------------------------------------------
# preact[j] = sum_i x[i] * ff[L][i][j].  We need column j contiguous.
# So transpose to [layer, j, i] and pack the i-axis as bits (256 bits = 32 B).
ff_np   = q["ff"].numpy()                   # [4, 256, 256]
ff_cols = ff_np.transpose(0, 2, 1)          # [4, j, i]
ff_packed = pack_pm1_bits(ff_cols)          # 4 * 256 * 32 = 32768 B

# ---------- head ------------------------------------------------------------
# logits[v] = sum_i read[i] * head[i][v].  Same trick: head[:, v] is what we
# XOR-popcount with the read vector.
head_T = q["head"].numpy().transpose(1, 0)  # [v, i]
head_packed = pack_pm1_bits(head_T)         # 128 * 16 = 2048 B

# ---------- thresholds ------------------------------------------------------
# In runtime we compute popcount(state ^ wts_col).  For 256-bit dot:
#   preact = 256 - 2 * popcount
#   bit = +1  iff  preact >= thresh
#       iff  popcount <= (256 - thresh) / 2
#       iff  popcount <  cmp_val   where cmp_val = floor((258 - thresh)/2)
ff_thresh = q["ff_thresh"].numpy().astype(np.int64)
print(f"threshold range: [{ff_thresh.min()}, {ff_thresh.max()}]")
cmp_vals = np.floor((258 - ff_thresh) / 2).astype(np.int64)
cmp_vals = np.clip(cmp_vals, 0, 257).astype(np.uint16)
thresh_bytes = cmp_vals.tobytes()           # little-endian, 4*256*2 = 2048 B

# ---------- popcount LUT ----------------------------------------------------
poplut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

# ---------- exp LUT for sampling at temperature 0.7 -------------------------
# weights[v]  ∝  exp(logit_v / T)
#             =  exp((128 - 2*pop_v) / (16*T))
#             ∝  exp(-2*(pop_v - pop_min) / (16*T))
# Indexed by (pop_v - pop_min) in [0, 128].  Scale to byte (max 255).
TEMP        = 0.7
LOGIT_SCALE = 1.0 / 16.0
factor      = -2.0 * LOGIT_SCALE / TEMP     # ≈ -0.1786

exp_lut = np.zeros(256, dtype=np.uint8)
for d in range(129):
    val = math.exp(factor * d)
    exp_lut[d] = max(0, min(255, int(round(val * 255))))

# ---------- write -----------------------------------------------------------
def write(name, data):
    with open(name, "wb") as f:
        f.write(bytes(data))
    print(f"  {name}: {len(data)} bytes")

write("poplut.bin",     poplut.tobytes())
write("explut.bin",     exp_lut.tobytes())
write("initial.bin",    initial_packed.tobytes())
write("embed.bin",      embed_packed.tobytes())
write("head.bin",       head_packed.tobytes())
write("ff_weights.bin", ff_packed.tobytes())
write("thresholds.bin", thresh_bytes)

print("done.")
