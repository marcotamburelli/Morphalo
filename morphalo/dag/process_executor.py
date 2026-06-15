import multiprocessing as mp
import queue
import signal
import threading
import time
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
    # SIGINT is handled only by the parent. The worker must finish the current
    # node, run post_run(), and release its CUDA context through normal process
    # teardown instead of being interrupted in the middle of CUDA work.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

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

    This executor runs node jobs in a separate worker process and optionally
    recycles that worker after a bounded number of CUDA-using node executions.
    The parent process keeps DAG scheduling state, while the worker owns the
    actual node execution, CUDA context, loaded model caches, and any runtime
    state created by Diffusers, PyTorch, or auxiliary model backends.

    The main purpose of this executor is to limit CUDA context lifetime without
    falling back to one process per node. A single worker can execute several
    CUDA nodes and reuse in-process model caches; once the configured CUDA-node
    budget is reached, the worker is stopped and the next CUDA node starts in a
    newly spawned process.

    Parameters
    ----------
    max_cuda_nodes : int
        Maximum number of CUDA-using node executions allowed in one worker
        process.

        When this limit is reached, graceful shutdown is requested before the
        next CUDA node is executed. Forced termination remains a fallback if the
        worker does not stop within the configured shutdown timeout. CPU-only
        nodes may continue to use the current worker after the limit has been
        reached; the restart is delayed until another CUDA node is about to run.

    cuda_cooldown_before_restart : float
        Number of seconds to wait before stopping a worker that reached the CUDA
        node limit.

        This delay gives the GPU, CUDA runtime, and driver a short settling
        window after the last CUDA-heavy operation in the chunk. It is a
        defensive mitigation for systems where immediate CUDA context teardown
        after sustained inference load appears to increase driver instability.

    cuda_cooldown_after_restart : float
        Number of seconds to wait after the old worker has been stopped and
        before the next worker is started.

        This delay separates CUDA context teardown from the next model load or
        ``.to('cuda')`` operation. It is intended to reduce rapid teardown/reload
        churn on fragile driver or firmware combinations.

    Notes
    -----
    This execution model is a mitigation for severe GPU instability observed
    after long sequences of otherwise successful CUDA inference calls. The
    failure does not resemble a normal PyTorch out-of-memory error or an
    application exception: it may occur near the end of a run or during CUDA
    teardown, with symptoms such as runaway GPU fans, loss of the GPU from the
    system, and the need for a forced machine restart.

    The working hypothesis is an NVIDIA driver or GSP firmware fragility
    triggered by long-lived CUDA contexts, repeated CUDA teardown/reload cycles,
    or a combination of both. Even when VRAM usage rises and falls normally, one
    context may accumulate internal activity from allocations, streams,
    cuDNN/cuBLAS handles, kernel caches, pipeline mutations, and repeated adapter
    or model loading and unloading. ``torch.cuda.empty_cache()`` releases unused
    cached allocations, but it does not destroy or recreate that context.

    The worker process owns the CUDA context and in-process model caches. Exiting
    the worker therefore gives the operating system and driver a complete process
    teardown boundary; the next CUDA node starts in a newly spawned process with
    a fresh context. Grouping several CUDA nodes in one worker preserves useful
    model-cache reuse while limiting context lifetime.

    The cooldown parameters are intentionally pragmatic. They do not fix CUDA,
    PyTorch, Diffusers, the NVIDIA driver, or GPU firmware. They only make worker
    recycling less abrupt, which may reduce the probability of triggering driver
    failures on unstable systems.

    ``SIGINT`` is owned by the parent process. The worker ignores it so a user
    interruption does not abruptly terminate an active CUDA call. On
    ``KeyboardInterrupt``, the parent stops scheduling new nodes, requests worker
    shutdown, and waits for the current node, its ``post_run()` hook, and normal
    process teardown to complete. Consequently, ``Ctrl+C`` is intentionally not
    immediate and may take as long as the current inference. Additional
    interrupts received while shutdown is in progress are deferred so they cannot
    break ``join()`` and leave the old CUDA worker alive while another pipeline is
    started.

    This is a defensive operational measure, not proof of the underlying driver
    cause and not a guarantee that every GPU or driver failure can be prevented.
    """

    def __init__(
        self,
        *,
        max_cuda_nodes: int = 8,
        cuda_cooldown_before_restart: float = 5.0,
        cuda_cooldown_after_restart: float = 3.0,
    ) -> None:
        """Initialize a process-based node executor.

        Parameters
        ----------
        max_cuda_nodes : int, default=8
            Maximum number of CUDA-using node executions allowed in one worker
            process.

            A value of ``1`` gives the strongest process isolation but discards
            model caches after every CUDA node and causes frequent CUDA context
            teardown/reload cycles. Higher values improve cache reuse and reduce
            restart churn, but keep the same CUDA context alive longer.

            The best value is workload- and driver-dependent. For the image
            generation workloads that motivated this executor, values in the
            ``8`` to ``24`` range are an experimental starting point rather than
            a general recommendation.

        cuda_cooldown_before_restart : float, default=5.0
            Seconds to wait before stopping a worker that reached
            ``max_cuda_nodes``.

            This delay is applied only when the executor is recycling a worker
            because the CUDA-node limit was reached. It is not applied to every
            node execution.

        cuda_cooldown_after_restart : float, default=3.0
            Seconds to wait after stopping the old worker and before allowing the
            next worker to be spawned.

            This delay is also applied only during CUDA-limit recycling, not
            during ordinary executor shutdown at the end of a run.

        Raises
        ------
        ValueError
            If ``max_cuda_nodes`` is not positive, or if either cooldown value is
            negative.

        Notes
        -----
        The executor uses the ``spawn`` multiprocessing start method. This avoids
        inheriting a potentially initialized CUDA runtime from the parent
        process, which would be unsafe with ``fork``-based process creation.

        The parent process evaluates ``node.uses_cuda`` to decide when to recycle
        the worker. That property must therefore be cheap and side-effect free:
        it should inspect node configuration only, without loading models,
        initializing CUDA, or touching GPU state.
        """

        if max_cuda_nodes <= 0:
            raise ValueError(
                f'max_cuda_nodes must be > 0, got {max_cuda_nodes}'
            )

        if cuda_cooldown_before_restart < 0:
            raise ValueError(
                'cuda_cooldown_before_restart must be >= 0, got '
                f'{cuda_cooldown_before_restart}'
            )

        if cuda_cooldown_after_restart < 0:
            raise ValueError(
                'cuda_cooldown_after_restart must be >= 0, got '
                f'{cuda_cooldown_after_restart}'
            )

        self.max_cuda_nodes = int(max_cuda_nodes)
        self.cuda_cooldown_before_restart = float(cuda_cooldown_before_restart)
        self.cuda_cooldown_after_restart = float(cuda_cooldown_after_restart)
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

    def close(
        self,
        *,
        graceful_timeout: Optional[float] = 30.0,
        terminate: bool = True,
    ) -> None:
        """Stop the current worker process.

        Parameters
        ----------
        graceful_timeout : float or None, default=30.0
            Maximum number of seconds to wait after requesting a graceful worker
            shutdown.

            If ``None``, wait indefinitely. This is useful after ``KeyboardInterrupt``
            because terminating a worker while it may be executing CUDA code can
            leave the driver or GPU runtime in a fragile state. In this mode,
            ``Ctrl+C`` does not cancel the active inference immediately: shutdown
            completes only after the worker finishes its current node,
            ``post_run()``, and normal process teardown.

        terminate : bool, default=True
            Whether to fall back to ``terminate()`` and then ``kill()`` if graceful
            shutdown does not complete within ``graceful_timeout``.

            Set this to ``False`` when handling user interruption and preferring a
            safe CUDA teardown over immediate process termination.
        """
        if self._process is None:
            return

        deferred_interrupt = False
        previous_sigint_handler = None

        if threading.current_thread() is threading.main_thread():
            previous_sigint_handler = signal.getsignal(signal.SIGINT)

            # Do not let repeated Ctrl+C presses interrupt process.join(). A
            # KeyboardInterrupt is raised only after the worker has stopped and
            # its queues and process handle have been closed.
            def defer_sigint(signum, frame) -> None:
                nonlocal deferred_interrupt
                deferred_interrupt = True
                print(
                    '[ProcessNodeExecutor] worker shutdown already in progress; '
                    'waiting for CUDA teardown to complete',
                    flush=True,
                )

            signal.signal(signal.SIGINT, defer_sigint)

        process = self._process
        job_queue = self._job_queue
        result_queue = self._result_queue

        try:
            if process.is_alive() and job_queue is not None:
                print(
                    f'[ProcessNodeExecutor] requesting worker stop pid={process.pid}',
                    flush=True,
                )
                job_queue.put(None)

                if graceful_timeout is None:
                    process.join()
                else:
                    process.join(timeout=graceful_timeout)

            if process.is_alive() and terminate:
                print(
                    f'[ProcessNodeExecutor] terminating worker pid={process.pid}',
                    flush=True,
                )
                process.terminate()
                process.join(timeout=10)

            if process.is_alive() and terminate:
                print(
                    f'[ProcessNodeExecutor] killing worker pid={process.pid}',
                    flush=True,
                )
                process.kill()
                process.join(timeout=10)

            if process.is_alive():
                raise ProcessNodeExecutionError(
                    f'Worker process {process.pid} did not stop; refusing to '
                    'start another worker before its resources are released'
                )

            print(
                f'[ProcessNodeExecutor] worker stopped pid={process.pid}',
                flush=True,
            )

            self._process = None
            self._job_queue = None
            self._result_queue = None
            self._cuda_count = 0

            self._close_queue(job_queue)
            self._close_queue(result_queue)
            process.close()
        finally:
            if previous_sigint_handler is not None:
                signal.signal(signal.SIGINT, previous_sigint_handler)

        if deferred_interrupt:
            raise KeyboardInterrupt

    def _restart_if_needed_before(self, node: NodeRef) -> None:
        if node.uses_cuda and self._cuda_count >= self.max_cuda_nodes:
            print(
                f'[ProcessNodeExecutor] restarting worker before node {node.id!r}: '
                f'cuda_count={self._cuda_count}/{self.max_cuda_nodes}, '
                f'cooldown_before={self.cuda_cooldown_before_restart}, '
                f'cooldown_after={self.cuda_cooldown_after_restart}',
                flush=True,
            )

            if self.cuda_cooldown_before_restart > 0:
                time.sleep(self.cuda_cooldown_before_restart)

            self.close()

            if self.cuda_cooldown_after_restart > 0:
                time.sleep(self.cuda_cooldown_after_restart)

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

        try:
            result = self._wait_for_result(node)
        except KeyboardInterrupt:
            print(
                '[ProcessNodeExecutor] interrupted; waiting for worker to finish current job '
                'and stop without terminate/kill',
                flush=True,
            )
            self.close(
                graceful_timeout=None,
                terminate=False,
            )
            raise

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
        if exc_type is not None and issubclass(exc_type, KeyboardInterrupt):
            self.close(
                graceful_timeout=None,
                terminate=False,
            )
        else:
            self.close()
