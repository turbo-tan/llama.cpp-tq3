# TQ3 Weight Compression

## Summary

`TQ3_1S` and `TQ3_4S` are custom GGUF weight formats developed for the
TurboQuant runtime. They target practical local inference: fit larger models
into smaller GPUs while keeping quality and speed good enough for real use.

The current preferred format is `TQ3_4S`.

## Why Another Format?

Standard GGUF formats such as `Q4_K_M`, `Q3_K_S`, and `IQ3_S` are mature and
portable. TurboQuant explores a different point in the design space:

- transform-assisted low-bit compression
- custom CUDA decode paths
- model-family-specific tensor policies
- stronger focus on prompt-processing throughput for compressed weights

The result is not a drop-in replacement for stock llama.cpp. It is a paired
format and runtime.

## Format Family

### `TQ3_1S`

`TQ3_1S` was the first compact TurboQuant target.

Its role:

- maximize compression
- prove the transform and custom runtime path
- expose which tensors cannot tolerate aggressive low-bit storage

Its limitation:

- quality tails were too large on some models
- one-scale metadata was not enough to consistently protect sensitive weight
  regions

### `TQ3_4S`

`TQ3_4S` is the quality-focused successor.

Its role:

- preserve the compact TQ3-style code path
- add more local scale flexibility
- reduce the KLD/perplexity tail seen in `TQ3_1S`
- become the default private release format

In practice, `TQ3_4S` trades some size for much better model behavior.

## Concrete `TQ3_4S` Block Layout

One `TQ3_4S` block stores `32` transformed weights:

```c
typedef struct {
    uint8_t d[4];
    uint8_t qs[12];
} block_tq3_4s;
```

That gives:

- `32` values per block
- `4` subgroups per block
- `8` values per subgroup
- `4` scale bytes per block
- `12` payload bytes per block
- `16` bytes total per block

The storage rate is:

```text
metadata = 4 scales * 8 bits = 32 bits
payload  = 32 indices * 3 bits = 96 bits
total    = 128 bits
bpw      = 128 / 32 = 4.00
```

For `N` quantized weights:

```text
size_bytes(TQ3_4S) = N * 4 / 8 = N / 2
```

ignoring GGUF tensor metadata and tensors intentionally left in other formats.

## Scale Format

Each `d[g]` is an unsigned 8-bit `E3M5` mini-float used as the scale for one
8-value subgroup:

```text
bits 7..5: exponent
bits 4..0: mantissa
```

Decode:

```c
if (byte == 0) {
    scale = 0;
} else {
    scale = 2^((byte >> 5) - 9) * (1 + (byte & 31) / 32)
}
```

This gives each subgroup a compact floating scale without storing four fp16
scales.

## Payload Packing

Each subgroup stores eight 3-bit centroid indices in three bytes:

```text
subgroup 0: qs[0..2]
subgroup 1: qs[3..5]
subgroup 2: qs[6..8]
subgroup 3: qs[9..11]
```

For one subgroup:

```text
idx0 =  qp[0]        & 7
idx1 = (qp[0] >> 3)  & 7
idx2 = ((qp[0] >> 6) | (qp[1] << 2)) & 7
idx3 = (qp[1] >> 1)  & 7
idx4 = (qp[1] >> 4)  & 7
idx5 = ((qp[1] >> 7) | (qp[2] << 1)) & 7
idx6 = (qp[2] >> 2)  & 7
idx7 = (qp[2] >> 5)  & 7
```

Each index selects one of eight fixed centroids.

## Mathematical Formulation

Let a 32-value source block be:

```text
w = [w_0, ..., w_31]
```

TurboQuant first applies a 32-point rotation / Hadamard-style transform:

```text
z = R(w)
```

where `R` is the fixed block transform used by the implementation.

The transformed vector is split into four 8-value subgroups:

```text
z_g = [z_{8g}, ..., z_{8g+7}],    g in {0,1,2,3}
```

Each transformed value is represented by:

```text
z_{g,j} ~= s_g * c_{q_{g,j}}
```

where:

- `s_g` is the subgroup scale
- `q_{g,j}` is a 3-bit index in `[0, 7]`
- `c_i` is the fixed Lloyd-Max centroid for index `i`

The codebook used by the current implementation is:

```text
C = [
  -1.996684, -1.291398, -0.740341, -0.247508,
   0.230106,  0.725222,  1.277503,  1.988943
]
```

These are 8-level Lloyd-Max centroids for an approximately standard-normal
rotated distribution. The reference code records theoretical centroid MSE
`0.03455` before scale fitting and inverse-transform effects.

The reconstructed block is:

```text
z_hat_{g,j} = s_g * c_{q_{g,j}}
w_hat       = R^-1(z_hat)
```

So the full `TQ3_4S` reconstruction equation is:

```text
w_hat_i = [R^-1( concat_g( s_g * C[q_g] ) )]_i
```

where `C[q_g]` means the 8 centroid lookups for subgroup `g`.

## Scale Fitting Objective

For each subgroup, the quantizer chooses indices and a scale that minimize
local squared error in the rotated domain:

```text
min_{s_g, q_{g,0..7}}  sum_{j=0}^{7} (z_{g,j} - s_g * C[q_{g,j}])^2
```

The implementation uses iterative fitting:

1. initialize `s_g` from subgroup RMS
2. assign each value to the nearest centroid after scaling
3. refit the scale with least squares
4. repeat for a small fixed number of iterations

Given fixed indices, the least-squares scale is:

```text
s_g = sum_j z_{g,j} * C[q_{g,j}] / sum_j C[q_{g,j}]^2
```

The fitted `s_g` is then encoded into the one-byte `E3M5` scale format.

## Dot-Product Equation

For a q8 activation block `a`, the ideal floating dot product over one TQ3 block
is:

```text
dot(w_hat, a) = sum_g sum_j (s_g * C[q_{g,j}]) * a_{g,j}
```

In the CUDA token-generation path, the centroids are mapped to int8 levels,
packed, and evaluated with `dp4a`:

```text
L = [-127, -79, -45, -14, 14, 45, 79, 127]
```

Two `dp4a` instructions cover one 8-value subgroup:

```text
I_g = dp4a(pack(L[q_{g,0..3}]), pack(a_{g,0..3}))
    + dp4a(pack(L[q_{g,4..7}]), pack(a_{g,4..7}))
```

The runtime rescales the integer dot back to the centroid domain:

```text
dot_g ~= I_g * s_g * (c_max / 127) * d_a
```

where:

- `c_max ~= 2.1519` in the token-generation approximation path
- `d_a` is the q8 activation scale

The prompt MMQ path uses a related idea but bakes subgroup scale ratios into
q8-style tile values before the MMQ consumer loop.

## Runtime Philosophy

The runtime should avoid paying the low-bit decode cost repeatedly.

Important wins have come from:

- decode-once paths for a tile
- `dp4a`-ready packing
- activation staging
- avoiding bit operations in the hot dot-product loop
- keeping correctness ahead of microbenchmark wins

The lesson from the kernel work is simple: an isolated CTA win is not enough.
Every kernel optimization must survive full prompt processing and chat tests.

## Quality Philosophy

TurboQuant quality work is measured by behavior, not only weight error.

The main gates are:

- perplexity on fixed corpora
- strict chat smoke tests
- code-generation style prompts
- long-context prompt processing
- model-specific comparisons against `Q3_K_S`, `Q4_K_M`, and available sources

This is important because low average error can still produce bad local tails.
The work from `TQ3_1S` to `TQ3_4S` was mostly about reducing those tails.

## Source Quality Rule

For new `TQ3_4S` model releases, prefer source models in this order:

1. `BF16` or `F16`
2. `Q8_0`
3. high-quality existing GGUF quant
4. already-low-bit GGUF or MLX quant, only if clearly labeled

Low-bit-to-low-bit conversion, such as `Q4_K_M -> TQ3_4S`, is requantization.
It can be useful for experiments, but it should not be presented as equivalent
to a clean `BF16/F16/Q8_0 -> TQ3_4S` conversion.

## Current Default

Use `TQ3_4S` for new TurboQuant model releases unless the deployment constraint
requires the smaller `TQ3_1S`.

Use `TQ3_1S` only when:

- the model otherwise does not fit
- quality loss is acceptable
- the release clearly labels it as a compact variant
