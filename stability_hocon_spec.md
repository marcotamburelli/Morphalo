## HOCON Input Specification

This document describes the **canonical structure of HOCON input files** used by Stability nodes (`t2i`, `i2i`, etc.).  
HOCON is used exclusively as a **human-friendly authoring DSL**. At runtime, configurations are parsed, resolved, and normalized into plain Python dictionaries.

---

### Top-level Structure

A job configuration may contain the following top-level keys:

```hocon
model { ... }
vae = "..."
prompt = "..."
negative_prompt = "..."
params { ... }
seed = random
```

All fields are optional unless stated otherwise, but some nodes require specific subsets.

---

## `model` (required)

Defines which checkpoint is used and how it is executed.

```hocon
model {
  kind   = sdxl_single_file
  path   = ~/models/juggernaut/juggernautXL_ragnarokBy.safetensors
  dtype  = bf16
  device = cuda
}
```

### Fields

- **`kind`**  
  Logical model type. Currently informational, but reserved for future dispatching.
  Typical value:
  - `sdxl_single_file`

- **`path`** *(required)*  
  Filesystem path to a `.safetensors` checkpoint.
  `~` is allowed and expanded at load time.

- **`dtype`** *(optional, default: `bf16`)*  
  Numerical precision used for inference.
  Allowed values:
  - `bf16`, `bfloat16`
  - `fp16`, `float16`
  - `fp32`, `float32`

- **`device`** *(optional, default: `cuda`)*  
  Execution device (`cuda` or `cpu`).

---

## `vae` (optional)

Overrides the VAE embedded in the checkpoint.

```hocon
vae = "madebyollin/sdxl-vae-fp16-fix"
```

If omitted, the checkpoint’s internal VAE is used.

---

## `prompt` (required)

Defines the **positive prompt**.

Accepted forms:

### String
```hocon
prompt = "cinematic portrait, soft rim light"
```

### Array (recommended)
```hocon
prompt = [
  ${character.base_prompt},
  """
  walking in a garden,
  cinematic lighting
  """
]
```

#### Normalization rules
- string → `strip()`
- array → elements joined with `".\n"`
- empty or missing → error (node-specific)

This allows modular prompt composition without relying on fragile HOCON string concatenation.

---

## `negative_prompt` (optional)

Defines constraints and exclusions.

Same accepted forms as `prompt`.

Typical usage:
```hocon
negative_prompt = [
  ${defaults.negative_prompt},
  "extra fingers",
  "deformed anatomy"
]
```

Normalization:
- array elements joined with `", "`

---

## `params` (optional)

Node-specific execution parameters.

### Common parameters

```hocon
params {
  steps  = 30
  cfg    = 6.0
  width  = 1024
  height = 1024
}
```

#### Meaning
- **`steps`**  
  Number of diffusion steps.

- **`cfg` / `guidance_scale`**  
  Classifier-free guidance scale.

- **`width`, `height`**  
  Output resolution (mostly relevant for `t2i`).

### Img2Img-specific
```hocon
params {
  strength = 0.7
}
```

- **`strength`**  
  Controls how strongly the input image is altered.
  - low (`0.2–0.4`) → refinement
  - high (`0.6–0.9`) → transformation

---

## `seed` (optional)

Controls randomness.

```hocon
seed = random
```
or
```hocon
seed = 123456789
```

- `random` → secure random seed generated at runtime
- integer → deterministic output

The effective seed is always returned in node output.

---

## Node Output (JSON)

Each node emits **one JSON object to stdout**.

Example:

```json
{
  "ok": true,
  "node": "t2i",
  "image": "outputs/2025-12-20_001122_t2i_seed123.png",
  "seed": 123,
  "params": {
    "steps": 30,
    "guidance_scale": 6.0
  },
  "timing": {
    "seconds": 2.84
  },
  "cuda_mem": {
    "allocated_gb": 3.21,
    "reserved_gb": 4.00,
    "peak_allocated_gb": 6.87
  },
  "metadata": "outputs/2025-12-20_001122_t2i_seed123.json"
}
```

### Output guarantees
- stdout contains **only JSON**
- logs go to stderr
- images and metadata are written to disk
- JSON is designed for piping between nodes

---

## Using Nodes and Pipes

Basic usage:

```bash
t2i job.conf
i2i job.conf input.png
```

Pipeline composition:

```bash
t2i job_t2i.conf | i2i job_i2i.conf
```

Semantics:
1. upstream node writes JSON
2. downstream node reads JSON from stdin
3. `image` field is used as input

This enables linear pipelines without temporary glue code.

---

## Design Rationale

- HOCON → expressive authoring
- Python dict → normalized execution
- JSON → stable node interface
- filesystem → image persistence

The system treats images as **side effects**, not control flow, enabling reproducible and scriptable pipelines.
