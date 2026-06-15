# Morphalo

Morphalo is a **local-first framework** for building reproducible **generative image and video workflows** as
**DAGs (Directed Acyclic Graphs)**.

Instead of exposing the internal components of diffusion models, Morphalo treats
complete diffusion pipelines as **high-level operators** that can be composed into
larger image-processing workflows.

Diffusers pipelines (`txt2img`, `img2img`, `inpaint`, foundation-model editing,
and experimental video generation) become building blocks inside a DAG,
alongside tools such as MediaPipe, Segment Anything, image preprocessors, and
scoring nodes.

The core idea is simple:

- You describe a pipeline as a graph of nodes (e.g. `Prompt` → `Txt2Img` → `Img2Img` → `Inpaint`).
- Each node materializes its output to disk (JSON sidecars + images), making runs **inspectable** and **restartable**.
- A lightweight runner executes the graph in dependency order and supports
  full, partial, and downstream runs.

Under the hood, Morphalo focuses on wiring together modern diffusion tooling
(Diffusers, ControlNet, IP-Adapter, FaceID, T2I-Adapter, Qwen Image, OmniGen,
and LTX-Video) with useful preprocessors and evaluators in a way that is
**composable**, **inspectable**, and **easy to iterate on**.

> **Status:** this repository is primarily intended for personal/local use for now. Expect breaking changes.

## Why this exists

Most diffusion experiments start as a notebook or a single script.  
As pipelines grow more complex, they often turn into:

- tangled glue code
- implicit dependencies between steps
- repeated preprocessing
- runs that are difficult to reproduce or partially rerun

Morphalo was created to address these issues while preserving the **fast iteration workflow** typical of experimental diffusion projects.

At its core, Morphalo is a **Python interface built on top of Diffusers** that allows complex image generation workflows to be expressed as **explicit DAGs (Directed Acyclic Graphs)**.

In practice, this makes Morphalo a **programmable pipeline engine for generative workflows**.

This approach introduces a few key improvements:

- **Explicit pipeline structure**  
  Dependencies between steps are expressed directly in the graph wiring.

- **Artifact-based execution**  
  Nodes materialize their outputs to disk, enabling inspection, incremental
  execution, and reuse of previously generated upstream artifacts.

- **Incremental execution**  
  Individual nodes or subgraphs can be re-executed without recomputing the entire pipeline.

- **Composable conditioning**  
  Multiple conditioning signals (prompt bundles, ControlNet, IP-Adapter, FaceID, etc.) can be injected laterally into generation nodes.

- **CLI-driven workflows**  
  DAGs can be executed, inspected, and partially rerun via a simple command-line interface.

- **Bounded CUDA worker lifetime**
  Nodes execute in a separate worker process. The worker is restarted after a
  configurable number of CUDA node executions to limit CUDA context lifetime.

### Relationship to existing tools

Several tools already exist for building diffusion workflows, most notably **ComfyUI**, which provides a graphical node-based interface.

Morphalo takes a different approach:

- **code-first rather than UI-first**
- designed for **reproducible experimentation**
- integrates naturally into **Python workflows and scripts**
- favors **explicit configuration and version-controlled pipelines**

While ComfyUI focuses on interactive visual experimentation, Morphalo focuses on **structured and programmable pipelines**.

In particular, Morphalo operates at a **higher abstraction level** than tools such as ComfyUI.

In ComfyUI, users typically construct graphs that closely mirror the **internal structure of diffusion inference**. 
Nodes often represent low-level components of the generation process (e.g. CLIP encoders, schedulers, samplers, VAE decoding, latent transformations), 
and building a workflow means assembling these internal building blocks manually.

Morphalo instead treats diffusion models as **high-level operators** by relying directly on ready-made pipelines provided by
libraries such as Diffusers. Rather than exposing the internal model graph, Morphalo orchestrates complete generation steps such as:

- `Txt2Img`
- `Img2Img`
- `Inpaint`
- ControlNet-conditioned generation
- adapter-based conditioning (IP-Adapter, FaceID, etc.)

These operators are then combined inside a DAG to build **larger image-processing workflows**.

This makes it natural to integrate diffusion with other processing stages, for example:

- preprocessing images (cropping subjects, extracting faces, generating control maps)
- running diffusion pipelines with prompts and conditioning signals
- refining or reprocessing generated outputs

In practice, a Morphalo workflow may combine components from several libraries—such as Diffusers, MediaPipe, Segment Anything (SAM), 
or other preprocessing tools—to construct end-to-end generation pipelines driven by prompts, images, or both.

The result is a system where diffusion pipelines become **building blocks in a broader image-processing DAG**, 
rather than the graph itself.

That said, the DAG model used by Morphalo is intentionally **UI-friendly**.  
Because workflows are represented as graphs with explicit channels, it would be straightforward in the future to build a graphical editor that generates DAG definitions automatically.

## Features (high level)

- **DAG-based pipeline authoring**  
  Workflows are defined as directed acyclic graphs where each node produces
  materialized artifacts. Nodes are connected via a lightweight wiring DSL
  (`NodeRef`, `AttachmentSink`, and channel-based connections).

- **Dependency-ordered DAG execution via CLI**
  A CLI runner discovers DAGs at module import time and executes nodes according
  to their declared dependencies. The runner supports full DAG execution as
  well as partial runs (single node, downstream, or forced upstream).

- **Artifact-based execution model**  
  Each node materializes its outputs to disk (images, video frames, metadata).
  JSON sidecar artifacts act as persistent checkpoints, enabling resumable
  execution and reuse of upstream outputs without recomputation.

- **Multi-channel wiring model**  
  Nodes communicate through explicit channels rather than positional inputs.
  This allows multiple forms of conditioning to be injected laterally into
  generation nodes.

  Typical lateral inputs include:

  - **Prompt bundles** injected into generation nodes
  - **ControlNet conditioning images**
  - **IP-Adapter slots** (optionally with spatial masks)
  - **FaceID conditioning** (optionally with masks and CLIP embedding injection)
  - **T2I-Adapter attachments**

- **Prompt pipeline**  
  Prompts are first-class nodes. They can be composed from structured content
  and style components, optionally translated to English, and injected into
  downstream generation nodes via a dedicated prompt channel.

- **Preprocessing nodes**  
  Several nodes prepare conditioning data for downstream models, including:
  - auxiliary maps (e.g. Canny, depth, pose)
  - subject- and face-aware cropping
  - prompt-guided object extraction with `AnyCrop`
  - basic image transformations such as resize, transpose, and flip
  - image compositing and stacking
  - conditioning preparation for adapters.

- **Image generation nodes**  
  Support for common diffusion workflows such as:

  - `txt2img`
  - `img2img`
  - `inpaint`

  with optional ControlNet, IP-Adapter, FaceID, and T2I-Adapter conditioning.

- **Foundation image nodes**
  Higher-level image generation and editing workflows are available through:

  - `QwenImage`
  - `QwenImageEdit`
  - `QwenImageEditPlus`
  - `QwenImageInpaint`
  - `OmniGen`

  These are large models with substantial memory and runtime requirements.
  Although their pipelines use CUDA-aware dispatch strategies such as
  `device_map='balanced'`, long or complex workloads may still expose GPU
  instability on consumer graphics cards.

- **Image evaluation nodes**
  Candidate images can be ranked using composable person, face, identity, and
  prompt-alignment scoring stages. This remains an experimental feature and is
  not a replacement for human evaluation. In many workflows, the more reliable
  approach is still to rerun a generation node with `--node` and manually select
  the best result.

- **Experimental video generation nodes**  
  Early support for video workflows is included (still under development):

  - `txt2video`
  - `img2video`

  These nodes integrate emerging video diffusion pipelines and will evolve as
  the ecosystem matures.

## Conceptual model

A Morphalo workflow is defined as a **directed acyclic graph (DAG)**.

Each node produces artifacts on disk and can receive inputs through
explicit channels. Nodes are connected using a wiring DSL that allows both
**main data flow** and **lateral conditioning inputs**.

### Node model

In Morphalo, each step of a workflow is represented by a **node**.

A node is a lightweight object that:

- has a unique **name** within the DAG
- receives a **spec** (configuration)
- optionally consumes artifacts produced by upstream nodes
- materializes its outputs to disk
- declares whether its configured execution may use CUDA
- may perform post-run synchronization or cleanup

Conceptually:

```
Node
 ├ name        (identifier within the DAG)
 ├ spec        (configuration dictionary or layered spec)
 ├ uses_cuda   (runtime capability derived from configuration)
 ├ run(...)    (execution logic)
 ├ post_run()  (optional synchronization or cleanup hook)
 └ artifacts   (files written to disk)
```

The `run(...)` method is responsible for executing the node's logic and
producing its artifacts. These artifacts are later consumed by downstream
nodes through the DAG wiring system.

Node objects are sent to a worker process before execution. Custom nodes should
therefore keep their declared state serializable and create heavyweight runtime
objects such as model pipelines, adapters, and device resources lazily during
`run()`.

`uses_cuda` defaults to `False`. Nodes that may execute CUDA work override it
and derive the result from their resolved configuration. CUDA-aware nodes use
`post_run()` to synchronize pending work before the runner proceeds or begins
resource cleanup.

This design makes pipeline execution explicit and inspectable while preserving
the ability to rerun only selected parts of a workflow.

A typical graph might look like this:

```mermaid
flowchart LR

Prompt --> Txt2Img
ControlNet["ControlNet (canny)"] --> Txt2Img
IPAdapter["IP-Adapter (style)"] --> Txt2Img
FaceID["FaceID (identity)"] --> Txt2Img

Txt2Img --> Img2Img
Img2Img --> Output
```

### Main flow

The **main flow** represents the primary data transformation.

Example:

```mermaid
flowchart LR

Txt2Img --> Img2Img --> Output
```

Each node consumes the artifacts produced by the previous node.

### Lateral wiring (conditioning)

Generation nodes can also receive **lateral inputs** through dedicated
channels.

These inputs do not replace the main image flow but **augment the generation
process**.

Typical examples:

- **Prompt → generation node**  
  Injects the resolved prompt bundle into the diffusion pipeline.

- **ControlNet → generation node**  
  Provides structural guidance (e.g, depth, canny edges).

- **IP-Adapter → generation node**  
  Injects style or reference-image conditioning.

- **FaceID → generation node**  
  Injects identity embeddings extracted from reference faces.

Example with lateral inputs:

```mermaid
flowchart LR

%% main pipeline
Txt2Img --> Img2Img --> Output

%% lateral conditioning
Prompt -.-> Txt2Img
ControlNet["ControlNet (canny)"] -.-> Txt2Img
IPAdapter["IP-Adapter (style)"] -.-> Txt2Img
FaceID["FaceID (identity)"] -.-> Txt2Img

%% styling
classDef main fill:#e6f2ff,stroke:#2b6cb0,stroke-width:2px
classDef cond fill:#fff5e6,stroke:#d69e2e,stroke-width:1px

class Txt2Img,Img2Img,Output main
class Prompt,ControlNet,IPAdapter,FaceID cond
```

> Solid edges represent the main artifact flow, while dashed edges represent lateral conditioning inputs.

### Wiring API

Nodes are connected using a small wiring DSL built around the `>>` operator.

Two main patterns exist:

**Direct wiring (main artifact flow):**

```
upstream_node >> downstream_node
```

This connects the primary artifact produced by `upstream_node` to the main
input of `downstream_node`.

**Channel wiring (lateral inputs):**

```
conditioning_node >> target_node.channel()
```

Here the upstream node is attached to a specific **input channel** exposed by
the downstream node.

These channels are implemented internally through `AttachmentSink` objects
and allow nodes to expose structured attachment points for conditioning
signals such as prompts, ControlNet inputs, or adapter references.

This distinction keeps the **main data flow explicit**, while allowing
multiple optional conditioning signals to be injected without breaking the
pipeline structure.


### Artifact-based execution

Each node writes its outputs to disk using a node-local artifact directory:

```
outputs/
  node_id/
    2026-06-13_142530_seed1234.png
    2026-06-13_142530_seed1234.json
```

The base directory is the `out_dir` supplied when the DAG is declared. Dotted
node identifiers are mapped to nested directories, so a node named
`refine.face.out` writes under `out_dir/refine/face/out/`.

JSON sidecars typically record:

- generation parameters and model information
- produced files
- timing and optional CUDA memory statistics
- conditioning metadata where applicable

These artifacts act as **persistent checkpoints**, allowing:

- partial DAG execution
- node-level reruns
- reuse of upstream results

without recomputing the entire pipeline.

When a partial run needs an upstream output, Morphalo loads the most recent
timestamped JSON sidecar in that node's artifact directory. This is deliberately
simple: artifacts are not automatically invalidated when code, model versions,
or node specifications change. The user decides when an upstream node should be
rerun.

## Repository layout (typical)

The repository is structured as a small execution engine plus optional local workspace directories.

### Core engine (version-controlled)

- `morphalo/` - Core Python package. Contains:
    - DAG engine and validation logic
    - CLI entrypoint
    - Node implementations (txt2img, img2img, inpaint, controlnet, ip_adapter, face_id, t2i_adapter, etc.)
    - Conditioning and wiring utilities

- `demo/` - Example DAGs and macro workflows. Useful starting points include:
    - `01_minimal_txt2img.py` for the smallest image-generation DAG
    - `10_foundation_minimal.py` for Qwen Image and OmniGen
    - `12_scorers.py` for candidate ranking and chained evaluators
    - `14_video_wip.py` for current LTX-Video workflows
    - `macro/` for reusable higher-level graph composition
- `tests/` - Unit tests (including DAG-level tests).
- `third_party/`
Wrapped or vendored utilities (e.g. controlnet aux processors, external helpers).
- `bin/` - Helper scripts (e.g. run_dag.sh).

Root-level scripts:

- `init_project.sh` – bootstrap local virtual environment
- `fetch_media_pipe.sh` – download MediaPipe task models
- `requirements.txt` – Python dependencies
- `config.ini` – configuration

## Prerequisites

### System
- Linux recommended (works anywhere Python + PyTorch works).
- Python 3 (`python3 -m venv` is used by the bootstrap script).
- NVIDIA GPU strongly recommended for diffusion workloads.

### Python dependencies
All Python deps are in `requirements.txt` and include `diffusers[torch]`, `transformers`, `pyhocon`, `typer`, `mediapipe`, `insightface`, `onnxruntime`, `ultralytics`, `segment-anything-py`, etc.

### Torch / CUDA
The project bootstrap installs PyTorch wheels with CUDA runtime **cu128**.  
If you want a different CUDA version (or CPU-only), you can edit `init_project.sh`.

## Quickstart (after cloning)

From the repo root:

```bash
# 1) create a local venv + install torch + requirements
./init_project.sh

# (optional) rebuild the venv from scratch
./init_project.sh --rebuild
```

This will:

- Create `./venv`
- Install PyTorch (CUDA-enabled wheel)
- Install all dependencies from requirements.txt
- Perform a non-blocking CUDA sanity check

### Activate the environment

You can activate the virtual environment manually:

```
source ./venv/bin/activate
```

Or use the provided helper script:
```
./start_env
```

## Optional: download extra model assets

Some nodes rely on external assets such as MediaPipe task files. Most Hugging Face
models, including SAM/SAM-HQ backends used by `SubjectCrop`, are downloaded
automatically by Transformers on first use.

Helper scripts are provided:

```bash
# MediaPipe task files (pose/hand/face landmarker)
./fetch_media_pipe.sh
```

- MediaPipe models are placed under `~/models/mediapipe`.

## Running DAGs

### CLI entrypoint

The main CLI is `morphalo.cli`. The most common command is:

```bash
python -m morphalo.cli run-dags <module> \
    [--dag <name>] \
    [--node <id>] \
    [--downstream] \
    [--force-upstream] \
    [--cuda-chunk-size <count>]
```

The runner discovers DAGs **at import time** via `DagRegistry`, then executes
their nodes in dependency order.

The CLI also provides a small configuration inspection command:

```bash
python -m morphalo.cli dump path/to/spec.conf
```

This parses a HOCON specification and prints the resolved configuration as JSON.

### Convenience script

There is a helper script that ensures the local venv is activated and calls the CLI:

```bash
./bin/run_dag.sh <dag-module>
```

Example:

```bash
./bin/run_dag.sh dags.demo_img
```

This script calls `python -m morphalo.cli run-dags "$DAG_MODULE" "$@"`.

## Example: a simple txt2img DAG

Below is a minimal DAG you can drop into `dags/hello_txt2img.py`:

```python
from pathlib import Path

from morphalo.dag import DAG
from morphalo.nodes.prompt import Prompt
from morphalo.nodes.txt2img import Txt2Img

ROOT = Path(__file__).resolve().parents[1]

with DAG('hello_txt2img', out_dir=ROOT / 'outputs' / 'hello_txt2img') as dag:
    prompt = Prompt(
        name='prompt',
        spec={
            'prompt': {
                'content': [
                    'A cinematic portrait of a silver-haired mage with violet eyes.',
                    'Realistic skin texture, natural lighting.'
                ],
                'style': [
                    '35mm film look, shallow depth of field'
                ]
            },
            'negative_prompt': [
                'low quality', 'blurry'
            ],
        },
    )

    out = Txt2Img(
        name='out',
        spec={
            'model': {'id': 'stabilityai/stable-diffusion-xl-base-1.0'},
            'params': {'steps': 25, 'cfg': 5.0, 'width': 1024, 'height': 1024},
            'seed': 'random',
        },
    )

    prompt >> out.prompt()
```

Run it:

```bash
./bin/run_dag.sh dags.hello_txt2img --dag hello_txt2img
```

Outputs are written under `outputs/hello_txt2img/` (image file + JSON metadata per node).

### Direct connections vs lateral wiring

Morphalo distinguishes between two kinds of node connections.

#### Direct connection (main artifact flow)

A direct connection passes the primary output artifact of one node into
the main input of another node.

Example:

``` python
file_img >> img2img
```

Here, `img2img` receives the upstream image as its required main input.
This is the standard data flow of the DAG: one node produces the
artifact, the next node consumes it.

Use a direct connection when the downstream node requires that artifact
in order to run.

Typical examples include:

-   image source → `Img2Img`
-   image source → `Inpaint`
-   `Txt2Img` → `Img2Img`
-   `Img2Img` → `ImageStack`

#### Lateral wiring (conditioning channels)

Some nodes also expose dedicated attachment points for lateral
conditioning. These are not the main input of the node, but additional
signals that modify or guide its behavior.

Example:

``` python
prompt >> out.prompt()
```

Here, the `Prompt` node is not connected to the main input of `Txt2Img`.
Instead, it is injected through a dedicated prompt channel exposed by
`out.prompt()`.

This same pattern is used for other conditioning mechanisms such as:

-   prompt bundles
-   ControlNet inputs
-   IP-Adapter reference images
-   FaceID inputs
-   T2I-Adapter inputs

In other words:

-   **required primary data** should be passed through a direct
    connection
-   **conditioning inputs** should be attached through the node's
    dedicated lateral wiring API

For more advanced wiring patterns, see the examples in the `demo/`
directory.

## Node specification resolution

Nodes accept their configuration through the `spec` parameter.

A spec can be provided in three forms:

-   a Python dictionary
-   a path to a HOCON configuration file
-   a sequence of layered specifications combining both

Example:

``` python
spec = [
    "configs/base.conf",
    "configs/scene.conf",
    {"params": {"steps": 30}}
]
```

During execution, Morphalo resolves this specification into a single
configuration dictionary.

Resolution works as follows:

1.  Each layer is resolved independently.
    -   dictionaries are used directly
    -   strings or `Path` objects are interpreted as paths to HOCON
        files and loaded accordingly
2.  The resolved layers are then **merged from left to right**.

Later layers override earlier ones.

### Merge rules

The merge operation follows a deterministic deep-merge strategy:

-   **nested dictionaries are merged recursively**
-   **scalars override previous values**
-   **lists are replaced entirely**

Example:

Layer 1:

``` python
{
  "params": {
    "steps": 20,
    "width": 1024
  }
}
```

Layer 2:

``` python
{
  "params": {
    "steps": 30
  }
}
```

Final result:

``` python
{
  "params": {
    "steps": 30,
    "width": 1024
  }
}
```

However, lists are not merged:

Layer 1:

``` python
{
  "negative_prompt": ["blurry"]
}
```

Layer 2:

``` python
{
  "negative_prompt": ["low quality"]
}
```

Final result:

``` python
{
  "negative_prompt": ["low quality"]
}
```

This layered specification model allows base configurations to be reused
and selectively overridden for different nodes or experiments while
keeping configuration files compact and composable.

## CLI Usage and Execution Model

Morphalo is designed around a small but flexible CLI runner that executes DAGs
in dependency order and supports partial execution.

The main entrypoint is:

```bash
python -m morphalo.cli run-dags <module> [options]
```

Where:

- `<module>` is a Python module that defines one or more DAGs.
- DAGs are discovered at import time via the internal registry.
- Execution is handled by the DAG runner.

### Running a full DAG

If a module defines multiple DAGs, you can select one explicitly:

```bash
./bin/run_dag.sh my_dags.some_pipeline --dag my_dag_name
```

If `--dag` is omitted, a full run executes all DAGs registered by the module in
definition order. When `--node` is used and the module contains multiple DAGs,
`--dag` is required to identify the target graph.

### Partial execution (node-level runs)

The CLI supports Make-like execution semantics.

You can run a single node by ID:

```bash
python -m morphalo.cli run-dags my_dags.some_pipeline \
    --dag my_dag_name \
    --node out
```

This requires that all upstream nodes have already materialized their outputs on disk.

If upstream artifacts are missing, you can force their execution:

```bash
python -m morphalo.cli run-dags my_dags.some_pipeline \
    --dag my_dag_name \
    --node out \
    --force-upstream
```

You can also execute a node and everything downstream from it:

```bash
python -m morphalo.cli run-dags my_dags.some_pipeline \
    --dag my_dag_name \
    --node some_node \
    --downstream \
    --force-upstream
```

This makes it possible to:
- modify one node,
- re-run only what is necessary,
- avoid recomputing expensive preprocessing steps.

### CUDA worker chunking

All node execution performed by `DAGRunner` takes place in a spawned worker
process. By default, the worker is restarted before the next CUDA node after it
has executed eight CUDA nodes:

```bash
python -m morphalo.cli run-dags my_dags.some_pipeline \
    --cuda-chunk-size 8
```

`--cuda-chunk-size` must be greater than zero. CPU nodes may continue in the
current worker after the threshold is reached; worker recycling is initiated
before the next CUDA node, with the cooldowns described below.

This mechanism limits the lifetime of a CUDA context. It was introduced as a
defensive mitigation for severe driver/GSP-level instability observed after
long and complex sequences of otherwise successful CUDA inference calls. It is
not an out-of-memory recovery mechanism and does not prove that a long-lived
context is the underlying cause.

Before a worker is replaced, the runner waits for the previous process to exit.
CUDA-aware nodes also complete pending GPU work before resources are cleaned up
or the worker is replaced.

> **Note:** Worker recycling includes a short pause around process replacement
> so GPU teardown and the next model load do not happen back-to-back. `Ctrl+C`
> also performs an orderly stop: Morphalo waits for the node currently being
> executed to finish before closing its worker. The interruption may therefore
> not be immediate, but avoids abandoning active GPU work in an uncertain state.

### Worker-process contract

Before each node is executed, Morphalo transfers the node and its resolved inputs
to the worker process. The worker performs the operation, completes any required
resource cleanup, and returns the resulting output to the DAG runner.

Because execution crosses a process boundary, custom nodes and their inputs must
be transferable between processes, and outputs must use serializable dictionary
structures. Runtime changes made inside the worker are local to that process and
do not update the original node object held by the DAG runner.

The worker process owns its CUDA context and model caches. Restarting it also
discards those process-local resources, trading some cache reuse for a shorter
CUDA context lifetime.

> **Note:** GPU models and their supporting libraries are loaded only when needed
> inside the worker. This keeps CUDA activity out of the parent process and
> preserves a clear ownership boundary: the DAG runner orchestrates execution,
> while the worker owns GPU resources.

## Artifact Model (Caching & Checkpointing)

Each successful node execution returns a dictionary and normally materializes
its files and JSON metadata under the DAG's `out_dir`. These sidecars serve as
persistent checkpoints for partial graph execution and upstream reuse.

Morphalo does not currently maintain a content-addressed cache or automatically
compare stored metadata against the current code and specification. A cached
artifact means "the most recent available output for this node id", not
"guaranteed valid for the current source tree".

Fixed seeds and explicit specifications improve repeatability, but exact output
reproduction may still depend on model versions, downloaded weights, PyTorch,
CUDA kernels, and other runtime details.

## Notes

- On first execution of a model, Hugging Face / Diffusers may download weights automatically.
- For best performance, use a CUDA-enabled GPU and appropriate dtype (fp16/bf16 where supported).

## License

Morphalo is licensed under the [Apache License 2.0](LICENSE).
Copyright 2026 Marco Tamburelli.

Third-party components under `third_party/` remain subject to their respective
licenses and attribution notices. Model weights and other externally downloaded
assets are distributed separately and may have additional license terms.
