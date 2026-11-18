# InstantHMR inference

The Medium backbone and transformer in `model.py` are adapted from
[ComfyUI-InstantHMR](https://github.com/jtydhr88/ComfyUI-InstantHMR/blob/ef8185aac9f9fe60b8898a4a8b9ad20263920916/instanthmr_model.py)
under Apache-2.0. The port uses standard PyTorch layers and keeps the pose and
camera inference required to render the body. The root LICENSE contains the
Apache-2.0 text.

`parameters.py` adapts the pose layout in [InstantHMR](https://github.com/Naturalpad-studio/InstantHMR/blob/main/training/mhr_cont.py),
which credits Meta's SAM 3D Body implementation. The parameter decoder and
Medium-commercial weights are covered by the [SAM License](LICENSE-SAM).

Morphalo fetches the default [Medium-commercial checkpoint](https://huggingface.co/jtydhr88/ComfyUI-InstantHMR)
through the Hugging Face cache. `fetch_mhr.sh` installs the separate
[MHR v1.0.1 TorchScript rig](https://github.com/facebookresearch/MHR/releases/tag/v1.0.1)
under `~/models/mhr`. The rig assets are under Apache-2.0.
