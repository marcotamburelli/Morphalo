import multiprocessing as mp
import queue
import traceback
from dataclasses import dataclass
from multiprocessing.reduction import ForkingPickler
from typing import Any, Dict, Optional

from morphalo.dag import NodeRef

Output = Dict[str, Any]


@dataclass
class NodeJob:
    node: NodeRef
    out_dir: str
    input_map: Dict[str, Output]


@dataclass
class NodeResult:
    ok: bool
    output: Optional[Output] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    traceback_text: Optional[str] = None


class ProcessNodeExecutionError(RuntimeError):
    """Raised when a node cannot be executed by the worker process."""


def _node_worker_main(
    job_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    while True:
        job = job_queue.get()

        if job is None:
            break

        try:
            out = job.node.run(
                job.out_dir,
                input=job.input_map,
            )

            job.node.post_run()

            if out is None:
                raise RuntimeError(f'No output for node {job.node.id!r}')

            result = NodeResult(
                ok=True,
                output=out,
            )

            try:
                ForkingPickler.dumps(result)
            except BaseException as exc:
                result = NodeResult(
                    ok=False,
                    error_type=type(exc).__name__,
                    error_message=(
                        f'Cannot serialize output for node {job.node.id!r}: '
                        f'{exc}'
                    ),
                    traceback_text=traceback.format_exc(),
                )

            result_queue.put(result)

        except BaseException as exc:
            result_queue.put(NodeResult(
                ok=False,
                error_type=type(exc).__name__,
                error_message=str(exc),
                traceback_text=traceback.format_exc(),
            ))


class ProcessNodeExecutor:
    """Run DAG nodes inside a restartable worker process.

    Parameters
    ----------
    max_cuda_nodes : int
        Maximum number of CUDA-using node executions allowed in one worker
        process. When the limit is reached, the worker is gracefully stopped
        and a new one is created before the next CUDA node run. CPU nodes may
        continue to use the current worker after the limit has been reached.

    Notes
    -----
    This execution model is a mitigation for severe GPU instability observed
    after long sequences of otherwise successful CUDA inference calls. The
    failure does not resemble a normal PyTorch out-of-memory error or an
    application exception: it may occur near the end of a run or during CUDA
    teardown, with symptoms such as runaway GPU fans, loss of the GPU from the
    system, and the need for a forced machine restart.

    The working hypothesis is an NVIDIA driver or GSP firmware fragility
    triggered by a long-lived CUDA context. Even when VRAM usage rises and falls
    normally, one context may accumulate internal activity from allocations,
    streams, cuDNN/cuBLAS handles, kernel caches, pipeline mutations, and repeated
    adapter or model loading and unloading. ``torch.cuda.empty_cache()`` releases
    unused cached allocations, but it does not destroy or recreate that context.

    The worker process owns the CUDA context and in-process model caches. Exiting
    the worker therefore gives the operating system and driver a complete process
    teardown boundary; the next CUDA node starts in a newly spawned process with
    a fresh context. Grouping several CUDA nodes in one worker preserves useful
    model-cache reuse while limiting context lifetime.

    This is a defensive operational measure, not proof of the underlying driver
    cause and not a guarantee that every GPU or driver failure can be prevented.
    """

    def __init__(self, *, max_cuda_nodes: int = 8) -> None:
        if max_cuda_nodes <= 0:
            raise ValueError(
                f'max_cuda_nodes must be > 0, got {max_cuda_nodes}'
            )

        self.max_cuda_nodes = int(max_cuda_nodes)
        self._ctx = mp.get_context('spawn')
        self._job_queue: Optional[mp.Queue] = None
        self._result_queue: Optional[mp.Queue] = None
        self._process: Optional[mp.Process] = None
        self._cuda_count = 0

    def _ensure_started(self) -> None:
        if self._process is not None and self._process.is_alive():
            return

        self._job_queue = self._ctx.Queue()
        self._result_queue = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_node_worker_main,
            args=(self._job_queue, self._result_queue),
            daemon=False,
        )

        try:
            self._process.start()
        except BaseException:
            job_queue = self._job_queue
            result_queue = self._result_queue
            self._process = None
            self._job_queue = None
            self._result_queue = None
            self._close_queue(job_queue)
            self._close_queue(result_queue)
            raise

        self._cuda_count = 0

    def _close_queue(self, process_queue: Optional[mp.Queue]) -> None:
        if process_queue is None:
            return

        process_queue.close()
        process_queue.join_thread()

    def close(self) -> None:
        if self._process is None:
            return

        process = self._process
        job_queue = self._job_queue
        result_queue = self._result_queue

        if process.is_alive() and job_queue is not None:
            job_queue.put(None)
            process.join(timeout=30)

        if process.is_alive():
            process.terminate()
            process.join(timeout=10)

        if process.is_alive():
            process.kill()
            process.join(timeout=10)

        if process.is_alive():
            raise ProcessNodeExecutionError(
                f'Worker process {process.pid} did not stop; refusing to '
                'start another worker before its resources are released'
            )

        self._process = None
        self._job_queue = None
        self._result_queue = None
        self._cuda_count = 0

        self._close_queue(job_queue)
        self._close_queue(result_queue)
        process.close()

    def _restart_if_needed_before(self, node: NodeRef) -> None:
        if node.uses_cuda and self._cuda_count >= self.max_cuda_nodes:
            self.close()

    def _wait_for_result(self, node: NodeRef) -> NodeResult:
        assert self._result_queue is not None
        assert self._process is not None

        while True:
            try:
                return self._result_queue.get(timeout=0.1)
            except queue.Empty:
                if self._process.is_alive():
                    continue

                try:
                    return self._result_queue.get_nowait()
                except queue.Empty:
                    raise ProcessNodeExecutionError(
                        f'Worker exited with code {self._process.exitcode} '
                        f'while running node {node.id!r}'
                    ) from None

    def run(
        self,
        *,
        node: NodeRef,
        out_dir: str,
        input_map: Dict[str, Output],
    ) -> Output:
        self._restart_if_needed_before(node)

        job = NodeJob(
            node=node,
            out_dir=out_dir,
            input_map=input_map,
        )

        try:
            ForkingPickler.dumps(job)
        except BaseException as exc:
            raise ProcessNodeExecutionError(
                f'Cannot serialize node job {node.id!r}: '
                f'{type(exc).__name__}: {exc}'
            ) from exc

        self._ensure_started()

        assert self._job_queue is not None
        assert self._result_queue is not None
        assert self._process is not None

        self._job_queue.put(job)

        result = self._wait_for_result(node)

        if not result.ok:
            raise ProcessNodeExecutionError(
                f'Worker failed while running node {node.id!r}: '
                f'{result.error_type}: {result.error_message}\n'
                f'{result.traceback_text or ""}'
            )

        if node.uses_cuda:
            self._cuda_count += 1

        if result.output is None:
            raise ProcessNodeExecutionError(
                f'Worker returned no output for node {node.id!r}'
            )

        return result.output

    def __enter__(self) -> 'ProcessNodeExecutor':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
