### Notes on Large Diffusion Models (Qwen-Image)

With very large models such as `Qwen/Qwen-Image`, the most reliable way to avoid OOM/OOO issues **without degrading image quality** is to load the model **without quantization**, using `bfloat16` and a balanced device map:

```python
pipeline = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image",
    torch_dtype=torch.bfloat16,
    device_map="balanced",
)
```

This approach distributes model components across GPU and CPU as needed, preserving numerical stability and visual fidelity.

It is possible to force full GPU execution (`device_map="cuda"`) by applying quantization, for example using 4-bit `bitsandbytes`:

```python
quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
    },
    components_to_quantize=["text_encoder"],
)
```

However, in practice this setup leads to a **drastic degradation in image quality**, likely due to reduced precision in the text encoding and/or internal latent representations.

In short: quantization improves memory usage and speed, but for this model it is **not suitable when image quality is the primary goal**.
