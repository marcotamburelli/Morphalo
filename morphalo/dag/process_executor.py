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

# Internal CUDA shutdown heuristics. These values are intentionally not exposed
# as public node parameters: they define executor-level safety behavior, not DAG
# semantics.
_CUDA_IDLE_MAX_WAIT = 60.0
_CUDA_STARTUP_IDLE_MAX_WAIT = 20.0
_CUDA_IDLE_POLL_INTERVAL = 2.0
_CUDA_IDLE_POWER_MARGIN_WATTS = 25.0
_CUDA_IDLE_TEMPERATURE_MARGIN_CELSIUS = 12.0
_CUDA_IDLE_MIN_POWER_WATTS = 30.0
_CUDA_IDLE_MIN_TEMPERATURE_CELSIUS = 40.0
_CUDA_IDLE_SHUTDOWN_MIN_PSTATE = 5
_CUDA_IDLE_STARTUP_MIN_PSTATE = 5
_CUDA_IDLE_FALLBACK_POWER_WATTS = 80.0
_CUDA_IDLE_FALLBACK_TEMPERATURE_CELSIUS = 45.0
_CUDA_IDLE_BASELINE_MAX_TEMPERATURE_CELSIUS = 45.0


@dataclass
class NodeJob:
    node: NodeRef
    out_dir: str
    input_map: Dict[str, Output]


@dataclass
class WorkerShutdown:
    """Request graceful worker shutdown with optional CUDA quiesce.

    Parameters
    ----------
    wait_for_cuda_idle : bool, default=True
        Whether the worker should try to synchronize CUDA work and wait for the
        GPU to appear idle before exiting.

    min_wait : float, default=5.0
        Minimum number of seconds to wait after CUDA synchronization and cache
        cleanup before polling GPU state.

    max_wait : float, default=60.0
        Maximum total number of seconds spent in the shutdown idle wait.

    poll_interval : float, default=2.0
        Delay between consecutive NVML probes.

    max_power_watts : float, default=80.0
        Maximum GPU power draw considered quiet enough. In normal execution this
        value is computed by the parent from the first valid worker baseline plus a
        margin. The default is a fallback used when no valid baseline is available.

    max_temperature_celsius : float, default=45.0
        Maximum GPU temperature considered quiet enough. In normal execution this
        value is computed by the parent from the first valid worker baseline plus a
        margin. The default is a conservative fallback used when no valid baseline
        is available.

    min_pstate : int, default=5
        Minimum accepted NVML performance state. Lower values are higher
        performance states, so ``5`` accepts ``P5`` and colder / more idle
        states, while rejecting ``P0`` through ``P4``.

    Notes
    -----
    This command is consumed only by the worker process. The parent uses it to
    request an orderly shutdown, but does not import PyTorch, NVML, or any
    CUDA-related monitoring library itself.
    """

    wait_for_cuda_idle: bool = True
    min_wait: float = 5.0
    max_wait: float = _CUDA_IDLE_MAX_WAIT
    poll_interval: float = _CUDA_IDLE_POLL_INTERVAL
    max_power_watts: float = _CUDA_IDLE_FALLBACK_POWER_WATTS
    max_temperature_celsius: float = _CUDA_IDLE_FALLBACK_TEMPERATURE_CELSIUS
    min_pstate: int = _CUDA_IDLE_SHUTDOWN_MIN_PSTATE


@dataclass
class WorkerWaitForCudaIdle:
    """Request a best-effort CUDA idle wait before the first CUDA job.

    This command is sent by the parent only when the next job is CUDA and the
    current worker has not yet performed a startup idle wait. The worker remains
    alive after handling it and replies with :class:`WorkerCudaIdleReady`.
    """

    max_wait: float = _CUDA_STARTUP_IDLE_MAX_WAIT
    poll_interval: float = _CUDA_IDLE_POLL_INTERVAL
    max_power_watts: float = _CUDA_IDLE_FALLBACK_POWER_WATTS
    max_temperature_celsius: float = _CUDA_IDLE_FALLBACK_TEMPERATURE_CELSIUS
    min_pstate: int = _CUDA_IDLE_STARTUP_MIN_PSTATE


@dataclass
class WorkerCudaIdleReady:
    """Report completion of the pre-CUDA idle wait command."""

    idle_reached: bool = False
    baseline: Optional['NvidiaIdleBaseline'] = None
    error_message: Optional[str] = None


@dataclass
class NvidiaIdleBaseline:
    """Idle-like NVIDIA GPU state measured by a worker process.

    Parameters
    ----------
    temperature_celsius : float
        Current GPU temperature in Celsius.

    power_watts : float
        Current GPU power draw in watts.

    pstate : int
        NVML performance state. Lower values mean higher performance states:
        ``0`` is ``P0``, while larger values are progressively more idle.

    gpu_utilization_percent : int
        Current GPU utilization percentage.

    memory_used_mib : int
        Currently allocated GPU memory in MiB.
    """
    temperature_celsius: float
    power_watts: float
    pstate: int
    gpu_utilization_percent: int
    memory_used_mib: int


@dataclass
class WorkerStarted:
    """Report worker startup status to the parent process.

    Parameters
    ----------
    baseline : NvidiaIdleBaseline or None
        Baseline GPU state measured by the worker before running any node.
        ``None`` means the baseline could not be measured.

    error_message : str or None
        Optional non-fatal error message explaining why the baseline could not
        be measured.
    """
    baseline: Optional[NvidiaIdleBaseline] = None
    error_message: Optional[str] = None


@dataclass
class NodeResult:
    ok: bool
    output: Optional[Output] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    traceback_text: Optional[str] = None


class ProcessNodeExecutionError(RuntimeError):
    """Raised when a node cannot be executed by the worker process."""


def _wait_for_nvidia_idle_in_worker(
    *,
    max_wait: float,
    poll_interval: float,
    max_power_watts: float,
    max_temperature_celsius: float,
    min_pstate: int,
) -> WorkerCudaIdleReady:
    """Wait for the GPU to look idle before CUDA work starts.

    Returns
    -------
    WorkerCudaIdleReady
        Command result containing the final measured baseline, or a non-fatal
        error if NVML could not be used.

    Notes
    -----
    This wait runs inside the worker, before the first CUDA node in that worker.
    It does not import PyTorch or create a CUDA context; it uses NVML only as a
    best-effort sensor so a CUDA job does not immediately start loading models
    while the GPU still appears to be settling from the previous worker teardown.
    """
    try:
        from pynvml import (NVML_TEMPERATURE_GPU, NVMLError,
                            nvmlDeviceGetHandleByIndex,
                            nvmlDeviceGetMemoryInfo,
                            nvmlDeviceGetPerformanceState,
                            nvmlDeviceGetPowerUsage, nvmlDeviceGetTemperature,
                            nvmlDeviceGetUtilizationRates, nvmlInit,
                            nvmlShutdown)
    except BaseException as exc:
        return WorkerCudaIdleReady(
            error_message=f'NVML unavailable during pre-CUDA idle wait: {exc}',
        )

    deadline = time.monotonic() + max_wait
    last_baseline: Optional[NvidiaIdleBaseline] = None

    try:
        nvmlInit()
        try:
            handle = nvmlDeviceGetHandleByIndex(0)

            while True:
                try:
                    memory = nvmlDeviceGetMemoryInfo(handle)
                    utilization = nvmlDeviceGetUtilizationRates(handle)
                    temperature = nvmlDeviceGetTemperature(
                        handle,
                        NVML_TEMPERATURE_GPU,
                    )
                    power_watts = nvmlDeviceGetPowerUsage(handle) / 1000.0
                    pstate = nvmlDeviceGetPerformanceState(handle)

                    last_baseline = NvidiaIdleBaseline(
                        temperature_celsius=float(temperature),
                        power_watts=float(power_watts),
                        pstate=int(pstate),
                        gpu_utilization_percent=int(utilization.gpu),
                        memory_used_mib=int(memory.used // (1024 * 1024)),
                    )

                    print(
                        f'[ProcessNodeExecutor] worker pre-CUDA idle probe: '
                        f'temp={temperature}C, '
                        f'power={power_watts:.1f}W, '
                        f'pstate=P{pstate}, '
                        f'util={utilization.gpu}%, '
                        f'mem={last_baseline.memory_used_mib}MiB',
                        flush=True,
                    )

                    if (
                        utilization.gpu == 0
                        and power_watts <= max_power_watts
                        and temperature <= max_temperature_celsius
                        and pstate >= min_pstate
                    ):
                        return WorkerCudaIdleReady(
                            idle_reached=True,
                            baseline=last_baseline,
                        )

                except NVMLError as exc:
                    return WorkerCudaIdleReady(
                        error_message=f'NVML pre-CUDA idle probe failed: {exc}',
                    )

                if time.monotonic() >= deadline:
                    if last_baseline is not None:
                        print(
                            '[ProcessNodeExecutor] pre-CUDA idle wait timed out; '
                            'continuing with last measured GPU state',
                            flush=True,
                        )
                        return WorkerCudaIdleReady(
                            idle_reached=False,
                            baseline=last_baseline,
                            error_message='NVML pre-CUDA idle wait timed out',
                        )

                    return WorkerCudaIdleReady(
                        error_message='NVML pre-CUDA idle wait timed out before first reading',
                    )

                time.sleep(poll_interval)
        finally:
            nvmlShutdown()

    except BaseException as exc:
        return WorkerCudaIdleReady(
            error_message=f'NVML pre-CUDA idle wait failed: {exc}',
        )


def _wait_for_cuda_idle_in_worker(
    *,
    min_wait: float,
    max_wait: float,
    poll_interval: float,
    max_power_watts: float,
    max_temperature_celsius: float,
    min_pstate: int,
) -> None:
    """Quiesce CUDA work and wait for the GPU to appear idle.

    Parameters
    ----------
    min_wait : float
        Minimum number of seconds to wait after CUDA synchronization and cache
        cleanup before polling GPU state.

    max_wait : float
        Maximum total number of seconds spent in the idle wait. If the GPU does
        not satisfy the idle heuristic within this window, the function returns
        and allows worker shutdown to continue.

    poll_interval : float
        Delay between consecutive NVML probes.

    max_power_watts : float
        Maximum power draw considered quiet enough.

    max_temperature_celsius : float
        Maximum GPU temperature considered quiet enough.

    Notes
    -----
    This function runs inside the worker process, never in the parent process.
    It may import CUDA-related libraries because the worker already owns the
    CUDA context and NVIDIA device handles.

    The function first synchronizes pending CUDA work, triggers Python garbage
    collection, and asks PyTorch to release unused cached CUDA allocations. This
    does not destroy the CUDA context; the real context boundary is still the
    worker process exit.

    NVML is used only as a best-effort sensor. It helps avoid exiting the worker
    immediately after a CUDA-heavy chunk while the GPU still appears hot or busy.
    It does not clean up PyTorch, Diffusers, CUDA, UVM, or NVIDIA driver state.

    Failure to import or query NVML is not fatal. In that case, the worker still
    performs CUDA synchronization and the minimum wait, then continues with
    normal shutdown.
    """
    import gc

    try:
        import torch

        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
    except BaseException as exc:
        print(
            f'[ProcessNodeExecutor] CUDA quiesce skipped: {exc}',
            flush=True,
        )

    if min_wait > 0:
        time.sleep(min_wait)

    try:
        from pynvml import (NVML_TEMPERATURE_GPU, NVMLError,
                            nvmlDeviceGetHandleByIndex,
                            nvmlDeviceGetMemoryInfo,
                            nvmlDeviceGetPerformanceState,
                            nvmlDeviceGetPowerUsage, nvmlDeviceGetTemperature,
                            nvmlDeviceGetUtilizationRates, nvmlInit,
                            nvmlShutdown)

    except BaseException as exc:
        print(
            f'[ProcessNodeExecutor] NVML unavailable; idle polling skipped: {exc}',
            flush=True,
        )
        return

    deadline = time.monotonic() + max(0.0, max_wait - min_wait)

    try:
        nvmlInit()
        try:
            handle = nvmlDeviceGetHandleByIndex(0)

            while time.monotonic() < deadline:
                try:
                    memory = nvmlDeviceGetMemoryInfo(handle)
                    utilization = nvmlDeviceGetUtilizationRates(handle)
                    temperature = nvmlDeviceGetTemperature(
                        handle,
                        NVML_TEMPERATURE_GPU,
                    )
                    power_watts = nvmlDeviceGetPowerUsage(handle) / 1000.0
                    memory_used_mib = memory.used // (1024 * 1024)
                    pstate = nvmlDeviceGetPerformanceState(handle)

                    print(
                        f'[ProcessNodeExecutor] worker GPU idle probe: '
                        f'temp={temperature}C, '
                        f'power={power_watts:.1f}W, '
                        f'pstate=P{pstate}, '
                        f'util={utilization.gpu}%, '
                        f'mem={memory_used_mib}MiB',
                        flush=True,
                    )

                    if (
                        utilization.gpu == 0
                        and power_watts <= max_power_watts
                        and temperature <= max_temperature_celsius
                        and pstate >= min_pstate
                    ):
                        return

                except NVMLError as exc:
                    print(
                        f'[ProcessNodeExecutor] NVML probe failed: {exc}',
                        flush=True,
                    )
                    return

                time.sleep(poll_interval)
        finally:
            nvmlShutdown()

    except BaseException as exc:
        print(
            f'[ProcessNodeExecutor] NVML idle wait skipped: {exc}',
            flush=True,
        )


def _node_worker_main(
    job_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    from morphalo.nodes.common.env import setup_env

    setup_env()

    # SIGINT is handled only by the parent. The worker must finish the current
    # node, run post_run(), and release its CUDA context through normal process
    # teardown instead of being interrupted in the middle of CUDA work.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    result_queue.put(WorkerStarted())

    while True:
        job = job_queue.get()

        if isinstance(job, WorkerWaitForCudaIdle):
            result_queue.put(_wait_for_nvidia_idle_in_worker(
                max_wait=job.max_wait,
                poll_interval=job.poll_interval,
                max_power_watts=job.max_power_watts,
                max_temperature_celsius=job.max_temperature_celsius,
                min_pstate=job.min_pstate,
            ))
            continue

        if isinstance(job, WorkerShutdown):
            # Shutdown is also part of the CUDA lifecycle. The worker performs
            # CUDA quiesce before exiting so the parent can remain GPU-blind and
            # only wait for process termination.
            if job.wait_for_cuda_idle:
                _wait_for_cuda_idle_in_worker(
                    min_wait=job.min_wait,
                    max_wait=job.max_wait,
                    poll_interval=job.poll_interval,
                    max_power_watts=job.max_power_watts,
                    max_temperature_celsius=job.max_temperature_celsius,
                    min_pstate=job.min_pstate,
                )

            break

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

    cuda_shutdown_min_wait : float
        Minimum number of seconds the worker waits during CUDA shutdown quiesce
        before polling GPU state.

        When the CUDA-node limit is reached, the parent requests a graceful
        worker shutdown. The worker then synchronizes CUDA work, releases unused
        PyTorch CUDA cache, waits at least this many seconds, and optionally uses
        NVML to wait until the GPU appears quiet enough to exit.

    cuda_cooldown_after_restart : float
        Number of seconds the parent waits after the old worker has exited and
        before the next worker may be started.

        This delay is intentionally short. It separates process teardown from
        the next CUDA context creation, while the heavier pre-exit settling work
        is performed inside the worker.

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

    Worker recycling is intentionally conservative. Before exiting, the worker
    can synchronize CUDA work, release unused PyTorch CUDA cache, wait for a
    minimum settling window, and use NVML as a best-effort sensor to check
    whether the GPU appears idle enough. After the worker exits, the parent may
    still wait briefly before spawning the next worker.

    These waits do not fix CUDA, PyTorch, Diffusers, the NVIDIA driver, or GPU
    firmware. They only make CUDA context teardown and recreation less abrupt,
    which may reduce the probability of triggering driver failures on unstable
    systems.

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
        cuda_shutdown_min_wait: float = 5.0,
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

        cuda_shutdown_min_wait : float, default=5.0
            Minimum number of seconds used by the worker-side CUDA quiesce step
            before polling GPU state during graceful shutdown.

            This is not a parent-side sleep. The delay is applied inside the
            worker after CUDA synchronization and cache cleanup, before the optional
            NVML idle polling loop starts.

        cuda_cooldown_after_restart : float, default=3.0
            Seconds to wait after the old worker has exited and before allowing
            the next worker to be spawned.

            This is a small parent-side gap between process teardown and the next
            CUDA context creation. The main shutdown settling logic happens
            inside the worker.

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

        if cuda_shutdown_min_wait < 0:
            raise ValueError(
                'cuda_shutdown_min_wait must be >= 0, got '
                f'{cuda_shutdown_min_wait}'
            )

        if cuda_cooldown_after_restart < 0:
            raise ValueError(
                'cuda_cooldown_after_restart must be >= 0, got '
                f'{cuda_cooldown_after_restart}'
            )

        self.max_cuda_nodes = int(max_cuda_nodes)
        self.cuda_shutdown_min_wait = float(cuda_shutdown_min_wait)
        self.cuda_cooldown_after_restart = float(cuda_cooldown_after_restart)
        self._ctx = mp.get_context('spawn')
        self._job_queue: Optional[mp.Queue] = None
        self._result_queue: Optional[mp.Queue] = None
        self._process: Optional[mp.Process] = None
        self._cuda_count = 0
        self._worker_used_cuda = False
        self._worker_cuda_idle_wait_done = False
        self._cuda_idle_baseline: Optional[NvidiaIdleBaseline] = None
        self._cuda_idle_max_power_watts = _CUDA_IDLE_FALLBACK_POWER_WATTS
        self._cuda_idle_max_temperature_celsius = (
            _CUDA_IDLE_FALLBACK_TEMPERATURE_CELSIUS
        )
        self._cuda_idle_min_pstate = _CUDA_IDLE_SHUTDOWN_MIN_PSTATE

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
            self._read_worker_startup()
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
        self._worker_used_cuda = False
        self._worker_cuda_idle_wait_done = False

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

            For workers that executed CUDA nodes, the effective join timeout may
            be extended so worker-side idle waiting is not interrupted before its
            configured maximum duration.

        terminate : bool, default=True
            Whether to fall back to ``terminate()`` and then ``kill()`` if graceful
            shutdown does not complete within ``graceful_timeout``.

            Set this to ``False`` when handling user interruption and preferring a
            safe CUDA teardown over immediate process termination.

        Notes
        -----
        Graceful shutdown sends a :class:`WorkerShutdown` command instead of
        abruptly killing the worker. The worker may perform CUDA synchronization,
        cache cleanup, a minimum settling wait, and optional NVML polling before
        exiting. The parent does not read an explicit shutdown ACK; successful
        process termination is the acknowledgement.

        If the worker has not executed CUDA nodes, shutdown skips CUDA quiesce
        and returns immediately after the worker exits normally. CUDA idle
        waiting is used only for workers that actually ran CUDA work.
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
        join_timeout = graceful_timeout

        if graceful_timeout is not None and self._worker_used_cuda:
            join_timeout = max(
                graceful_timeout,
                _CUDA_IDLE_MAX_WAIT + 10.0,
            )

        try:
            if process.is_alive() and job_queue is not None:
                print(
                    f'[ProcessNodeExecutor] requesting worker stop pid={process.pid}',
                    flush=True,
                )
                job_queue.put(WorkerShutdown(
                    wait_for_cuda_idle=self._worker_used_cuda,
                    min_wait=self.cuda_shutdown_min_wait,
                    max_wait=_CUDA_IDLE_MAX_WAIT,
                    poll_interval=_CUDA_IDLE_POLL_INTERVAL,
                    max_power_watts=self._cuda_idle_max_power_watts,
                    max_temperature_celsius=self._cuda_idle_max_temperature_celsius,
                    min_pstate=self._cuda_idle_min_pstate,
                ))

                if join_timeout is None:
                    process.join()
                else:
                    process.join(timeout=join_timeout)

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
            self._worker_used_cuda = False
            self._worker_cuda_idle_wait_done = False

            self._close_queue(job_queue)
            self._close_queue(result_queue)
            process.close()
        finally:
            if previous_sigint_handler is not None:
                signal.signal(signal.SIGINT, previous_sigint_handler)

        if deferred_interrupt:
            raise KeyboardInterrupt

    def _restart_if_needed_before(self, node: NodeRef) -> None:
        force_cooldown = node.forces_cuda_cooldown
        limit_reached = node.uses_cuda and self._cuda_count >= self.max_cuda_nodes
        if (force_cooldown and self._process is not None) or limit_reached:
            print(
                f'[ProcessNodeExecutor] restarting worker before node {node.id!r}: '
                f'forced={force_cooldown}, '
                f'cuda_count={self._cuda_count}/{self.max_cuda_nodes}, '
                f'shutdown_min_wait={self.cuda_shutdown_min_wait}, '
                f'cooldown_after={self.cuda_cooldown_after_restart}',
                flush=True,
            )

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

    def _read_worker_startup(self) -> None:
        """Read the worker startup message."""
        assert self._result_queue is not None
        assert self._process is not None

        while True:
            try:
                message = self._result_queue.get(timeout=0.1)
                break
            except queue.Empty:
                if self._process.is_alive():
                    continue

                raise ProcessNodeExecutionError(
                    f'Worker exited with code {self._process.exitcode} '
                    'before reporting startup status'
                ) from None

        if not isinstance(message, WorkerStarted):
            raise ProcessNodeExecutionError(
                f'Worker returned unexpected startup message: '
                f'{type(message).__name__}'
            )

        if message.baseline is not None:
            self._configure_cuda_idle_thresholds(message.baseline)
            return

        if message.error_message:
            print(
                f'[ProcessNodeExecutor] {message.error_message}; '
                'using fallback CUDA idle thresholds',
                flush=True,
            )

    def _wait_for_worker_cuda_idle_before_first_cuda_node(self) -> None:
        """Ask the worker to wait for GPU idle before its first CUDA job."""
        assert self._job_queue is not None
        assert self._result_queue is not None
        assert self._process is not None

        if self._worker_cuda_idle_wait_done:
            return

        self._job_queue.put(WorkerWaitForCudaIdle(
            max_wait=_CUDA_STARTUP_IDLE_MAX_WAIT,
            poll_interval=_CUDA_IDLE_POLL_INTERVAL,
            max_power_watts=self._cuda_idle_max_power_watts,
            max_temperature_celsius=self._cuda_idle_max_temperature_celsius,
            min_pstate=_CUDA_IDLE_STARTUP_MIN_PSTATE,
        ))

        while True:
            try:
                message = self._result_queue.get(timeout=0.1)
                break
            except queue.Empty:
                if self._process.is_alive():
                    continue

                raise ProcessNodeExecutionError(
                    f'Worker exited with code {self._process.exitcode} '
                    'while waiting for pre-CUDA idle acknowledgement'
                ) from None

        if not isinstance(message, WorkerCudaIdleReady):
            raise ProcessNodeExecutionError(
                f'Worker returned unexpected pre-CUDA idle message: '
                f'{type(message).__name__}'
            )

        self._worker_cuda_idle_wait_done = True

        if message.idle_reached and message.baseline is not None:
            self._configure_cuda_idle_thresholds(message.baseline)
            return

        if message.error_message:
            print(
                f'[ProcessNodeExecutor] {message.error_message}; '
                'continuing with fallback CUDA idle thresholds',
                flush=True,
            )

    def _is_valid_cuda_idle_baseline(
        self,
        baseline: NvidiaIdleBaseline,
    ) -> bool:
        """Return whether a worker pre-CUDA probe looks idle enough for baseline use.

        Parameters
        ----------
        baseline : NvidiaIdleBaseline
            GPU state measured by a worker before running CUDA jobs.

        Returns
        -------
        bool
            ``True`` when the reading looks idle enough to be used as the reference
            state for later CUDA shutdown waits.

        Notes
        -----
        A pre-CUDA probe is useful only if it was collected while the GPU was already
        reasonably quiet. If the first worker starts while the GPU is still busy,
        hot, or in a high-performance state, using that reading as a baseline would
        relax all later shutdown thresholds.
        """
        return (
            baseline.gpu_utilization_percent == 0
            and baseline.pstate >= _CUDA_IDLE_STARTUP_MIN_PSTATE
            and baseline.temperature_celsius <= (
                _CUDA_IDLE_BASELINE_MAX_TEMPERATURE_CELSIUS
            )
        )

    def _configure_cuda_idle_thresholds(
        self,
        baseline: NvidiaIdleBaseline,
    ) -> None:
        """Configure CUDA idle thresholds from the first worker baseline.

        Parameters
        ----------
        baseline : NvidiaIdleBaseline
            Initial GPU state measured by a worker before running CUDA jobs.

        Notes
        -----
        The parent receives only numeric telemetry from the worker. It does not
        import NVML, PyTorch, or any CUDA-related monitoring library.

        The first valid baseline is kept for the whole executor lifetime. Later
        workers may start while the GPU is still warm, so their startup readings are
        intentionally not used to relax the thresholds.
        """
        if self._cuda_idle_baseline is not None:
            return

        if not self._is_valid_cuda_idle_baseline(baseline):
            print(
                f'[ProcessNodeExecutor] ignoring non-idle CUDA baseline: '
                f'temp={baseline.temperature_celsius:.1f}C, '
                f'power={baseline.power_watts:.1f}W, '
                f'pstate=P{baseline.pstate}, '
                f'util={baseline.gpu_utilization_percent}%, '
                f'mem={baseline.memory_used_mib}MiB; '
                'GPU does not look idle enough for baseline use; '
                'using fallback CUDA idle thresholds',
                flush=True,
            )
            return

        self._cuda_idle_baseline = baseline
        self._cuda_idle_max_power_watts = max(
            _CUDA_IDLE_MIN_POWER_WATTS,
            baseline.power_watts + _CUDA_IDLE_POWER_MARGIN_WATTS,
        )
        self._cuda_idle_max_temperature_celsius = max(
            _CUDA_IDLE_MIN_TEMPERATURE_CELSIUS,
            baseline.temperature_celsius + _CUDA_IDLE_TEMPERATURE_MARGIN_CELSIUS,
        )
        self._cuda_idle_min_pstate = _CUDA_IDLE_SHUTDOWN_MIN_PSTATE

        print(
            f'[ProcessNodeExecutor] CUDA idle baseline: '
            f'temp={baseline.temperature_celsius:.1f}C, '
            f'power={baseline.power_watts:.1f}W, '
            f'pstate=P{baseline.pstate}, '
            f'util={baseline.gpu_utilization_percent}%, '
            f'mem={baseline.memory_used_mib}MiB; '
            f'thresholds: '
            f'temp<={self._cuda_idle_max_temperature_celsius:.1f}C, '
            f'power<={self._cuda_idle_max_power_watts:.1f}W, '
            f'pstate>=P{self._cuda_idle_min_pstate}',
            flush=True,
        )

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

        if node.uses_cuda:
            self._wait_for_worker_cuda_idle_before_first_cuda_node()
            self._worker_used_cuda = True

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
