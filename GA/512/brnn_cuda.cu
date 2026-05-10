#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <mutex>


namespace {

constexpr int CARRY_DIM = 384;
constexpr int READ_DIM = 128;
constexpr int D_MODEL = 512;
constexpr int VOCAB = 128;

constexpr int D_MODEL_WORDS = D_MODEL / 32; // 16
constexpr int CARRY_WORDS = CARRY_DIM / 32; // 12
constexpr int READ_WORDS = READ_DIM / 32;   // 4

constexpr int THREADS = D_MODEL;            // 512
constexpr int WARPS = THREADS / 32;         // 16

// dot range for the head is [-READ_DIM, READ_DIM] = [-128, 128] (step 2),
// so a 257-entry LUT still covers it exactly.
__constant__ float EXP_DOT_LUT_V4[257];


void initialize_lut_once() {
    static std::once_flag flag;

    std::call_once(flag, []() {
        float lut[257];

        for (int dot = -READ_DIM; dot <= READ_DIM; ++dot) {
            lut[dot + READ_DIM] = std::exp(static_cast<float>(dot) * 0.0625f);
        }

        C10_CUDA_CHECK(cudaMemcpyToSymbol(
            EXP_DOT_LUT_V4,
            lut,
            sizeof(lut)
        ));
    });
}


__device__ __forceinline__ float warp_reduce_sum_f32(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}


// ff layout: [num_ff, D_MODEL_WORDS, D_MODEL] (word-major).
__device__ __forceinline__ int xnor_popcount_dot_dmodel_wmajor(
    const uint32_t* __restrict__ x_words,
    const uint32_t* __restrict__ w_word_major,
    int out
) {
    int pop = 0;

#pragma unroll
    for (int k = 0; k < D_MODEL_WORDS; ++k) {
        const uint32_t w = __ldg(w_word_major + k * D_MODEL + out);
        pop += __popc(~(x_words[k] ^ w));
    }

    return 2 * pop - D_MODEL;
}


// head layout: [READ_WORDS, VOCAB] (word-major).
__device__ __forceinline__ int xnor_popcount_dot_read_wmajor(
    const uint32_t* __restrict__ x_words,
    const uint32_t* __restrict__ w_word_major,
    int out
) {
    int pop = 0;

#pragma unroll
    for (int k = 0; k < READ_WORDS; ++k) {
        const uint32_t w = __ldg(w_word_major + k * VOCAB + out);
        pop += __popc(~(x_words[k] ^ w));
    }

    return 2 * pop - READ_DIM;
}


__global__ void brnn_forward_v4_kernel(
    const uint32_t* __restrict__ initial_packed, // [16]
    const uint32_t* __restrict__ embed_packed,   // [128, 4], row-major
    const uint32_t* __restrict__ ff_packed,      // [num_ff, 16, 512], word-major
    const uint32_t* __restrict__ head_packed,    // [4, 128], word-major
    const int64_t* __restrict__ tokens,          // [B, T]
    float* __restrict__ losses,                  // [B]
    int B,
    int T,
    int num_ff
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    if (b >= B) {
        return;
    }

    __shared__ uint32_t state_a[D_MODEL_WORDS];
    __shared__ uint32_t state_b[D_MODEL_WORDS];

    __shared__ float warp_sums[WARPS];
    __shared__ int target_dot_shared;

    if (tid < D_MODEL_WORDS) {
        state_a[tid] = __ldg(initial_packed + tid);
    }

    __syncthreads();

    float local_loss = 0.0f;

    for (int t = 0; t < T; ++t) {
        // ------------------------------------------------------------
        // x = sign(x @ ff[layer]) for each layer
        // ff_packed layout: [num_ff, 16, 512]
        // ------------------------------------------------------------
        uint32_t* in_words = state_a;
        uint32_t* out_words = state_b;

        for (int layer = 0; layer < num_ff; ++layer) {
            const uint32_t* layer_weights =
                ff_packed + layer * D_MODEL_WORDS * D_MODEL;

            const int dot = xnor_popcount_dot_dmodel_wmajor(
                in_words,
                layer_weights,
                tid
            );

            const bool positive = dot > 0;
            const uint32_t mask = __ballot_sync(0xffffffffu, positive);

            if (lane == 0) {
                out_words[warp] = mask;
            }

            __syncthreads();

            uint32_t* tmp = in_words;
            in_words = out_words;
            out_words = tmp;
        }

        const uint32_t* final_words = in_words;

        // ------------------------------------------------------------
        // Head + cross entropy.
        // logits_j = dot_j / 16
        // CE = log(sum_j exp(dot_j / 16)) - dot_target / 16
        // dot_j is in {-128, -126, ..., 128}.
        // ------------------------------------------------------------
        const int64_t target64 = __ldg(tokens + b * T + t);
        const int target = static_cast<int>(target64);

        if (tid == 0) {
            target_dot_shared = 0;
        }

        __syncthreads();

        float exp_part = 0.0f;

        if (tid < VOCAB) {
            const uint32_t* read_words = final_words + CARRY_WORDS;

            const int dot = xnor_popcount_dot_read_wmajor(
                read_words,
                head_packed,
                tid
            );

            exp_part = EXP_DOT_LUT_V4[dot + READ_DIM];

            if (tid == target) {
                target_dot_shared = dot;
            }
        }

        float sum = warp_reduce_sum_f32(exp_part);

        if (lane == 0) {
            warp_sums[warp] = sum;
        }

        __syncthreads();

        if (warp == 0) {
            float block_sum = lane < WARPS ? warp_sums[lane] : 0.0f;
            block_sum = warp_reduce_sum_f32(block_sum);

            if (lane == 0) {
                const float target_logit =
                    static_cast<float>(target_dot_shared) * 0.0625f;

                local_loss += logf(block_sum) - target_logit;
            }
        }

        __syncthreads();

        // ------------------------------------------------------------
        // x_next = concat(final[0:CARRY_DIM], embed[token])
        //        = first 12 words from final + 4 words from embed
        // ------------------------------------------------------------
        if (tid < CARRY_WORDS) {
            state_a[tid] = final_words[tid];
        }

        if (tid < READ_WORDS) {
            state_a[CARRY_WORDS + tid] =
                __ldg(embed_packed + target * READ_WORDS + tid);
        }

        __syncthreads();
    }

    if (tid == 0) {
        losses[b] = local_loss;
    }
}


#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INT32(x) TORCH_CHECK((x).scalar_type() == torch::kInt32, #x " must be int32")
#define CHECK_INT64(x) TORCH_CHECK((x).scalar_type() == torch::kInt64, #x " must be int64")


torch::Tensor forward_cuda_v4(
    torch::Tensor initial_packed,
    torch::Tensor embed_packed,
    torch::Tensor ff_packed_word_major,
    torch::Tensor head_packed_word_major,
    torch::Tensor tokens
) {
    CHECK_CUDA(initial_packed);
    CHECK_CUDA(embed_packed);
    CHECK_CUDA(ff_packed_word_major);
    CHECK_CUDA(head_packed_word_major);
    CHECK_CUDA(tokens);

    initial_packed = initial_packed.contiguous();
    embed_packed = embed_packed.contiguous();
    ff_packed_word_major = ff_packed_word_major.contiguous();
    head_packed_word_major = head_packed_word_major.contiguous();
    tokens = tokens.contiguous();

    CHECK_CONTIGUOUS(initial_packed);
    CHECK_CONTIGUOUS(embed_packed);
    CHECK_CONTIGUOUS(ff_packed_word_major);
    CHECK_CONTIGUOUS(head_packed_word_major);
    CHECK_CONTIGUOUS(tokens);

    CHECK_INT32(initial_packed);
    CHECK_INT32(embed_packed);
    CHECK_INT32(ff_packed_word_major);
    CHECK_INT32(head_packed_word_major);
    CHECK_INT64(tokens);

    TORCH_CHECK(initial_packed.numel() == D_MODEL_WORDS,
                "initial_packed must have 16 words");

    TORCH_CHECK(embed_packed.dim() == 2, "embed_packed must be rank 2");
    TORCH_CHECK(embed_packed.size(0) == VOCAB,
                "embed_packed shape must be [128, 4]");
    TORCH_CHECK(embed_packed.size(1) == READ_WORDS,
                "embed_packed shape must be [128, 4]");

    TORCH_CHECK(ff_packed_word_major.dim() == 3,
                "ff_packed_word_major must be rank 3");
    TORCH_CHECK(ff_packed_word_major.size(1) == D_MODEL_WORDS,
                "ff_packed_word_major shape must be [num_ff, 16, 512]");
    TORCH_CHECK(ff_packed_word_major.size(2) == D_MODEL,
                "ff_packed_word_major shape must be [num_ff, 16, 512]");

    TORCH_CHECK(head_packed_word_major.dim() == 2,
                "head_packed_word_major must be rank 2");
    TORCH_CHECK(head_packed_word_major.size(0) == READ_WORDS,
                "head_packed_word_major shape must be [4, 128]");
    TORCH_CHECK(head_packed_word_major.size(1) == VOCAB,
                "head_packed_word_major shape must be [4, 128]");

    TORCH_CHECK(tokens.dim() == 2, "tokens must have shape [B, T]");

    initialize_lut_once();

    const int B = static_cast<int>(tokens.size(0));
    const int T = static_cast<int>(tokens.size(1));
    const int num_ff = static_cast<int>(ff_packed_word_major.size(0));

    auto losses = torch::empty(
        {B},
        torch::TensorOptions()
            .device(tokens.device())
            .dtype(torch::kFloat32)
    );

    const dim3 blocks(B);
    const dim3 threads(THREADS);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    brnn_forward_v4_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(initial_packed.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(embed_packed.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(ff_packed_word_major.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(head_packed_word_major.data_ptr<int32_t>()),
        tokens.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        B,
        T,
        num_ff
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return losses;
}

} // namespace


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &forward_cuda_v4,
        "Fused binarized recurrent neural network forward pass v4 "
        "(D_MODEL=512, CARRY=384, READ=128)"
    );
}
