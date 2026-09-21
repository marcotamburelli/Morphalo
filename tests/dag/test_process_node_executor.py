from __future__ import annotations

import os
import queue
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pytest

import morphalo.dag.process_executor as process_executor_mod
from morphalo.dag import DAG, NodeRef
from morphalo.dag.control_nodes import CudaCooldown
from morphalo.dag.process_executor import (
    NodeResult,
    ProcessNodeExecutionError,
    ProcessNodeExecutor,
    WorkerCudaIdleReady,
    WorkerShutdown,
    WorkerWaitForCudaIdle,
)
from morphalo.dag.runner import DAGRunner
from tests.dag.nodes import SourceNode


@dataclass
class FileTracingNode(NodeRef):
    trace_path: str
    cuda: bool = False
    fail: bool = False

    @property
    def uses_cuda(self) -> bool:
        return self.cuda

    def run(
        self,
        output_dir,
        input: Dict[str, Dict] = None,
    ) -> Dict[str, Any]:
        with Path(self.trace_path).open('a', encoding='utf-8') as trace:
            trace.write(f'run,{self.id},{os.getpid()}\n')

        if self.fail:
            raise ValueError('planned failure')

        return {'node_id': self.id, 'pid': os.getpid()}

    def post_run(self) -> None:
        with Path(self.trace_path).open('a', encoding='utf-8') as trace:
            trace.write(f'post_run,{self.id},{os.getpid()}\n')


@dataclass
class SlowFileTracingNode(FileTracingNode):
    delay: float = 0.5

    def run(
        self,
        output_dir,
        input: Dict[str, Dict] = None,
    ) -> Dict[str, Any]:
        with Path(self.trace_path).open('a', encoding='utf-8') as trace:
            trace.write(f'run,{self.id},{os.getpid()}\n')

        time.sleep(self.delay)
        return {'node_id': self.id, 'pid': os.getpid()}


class FakeQueue:
    def __init__(self, get_items=None) -> None:
        self.items = []
        self.get_items = list(get_items or [])
        self.closed = False
        self.joined = False

    def put(self, item) -> None:
        self.items.append(item)

    def get(self, timeout=None):
        if not self.get_items:
            raise queue.Empty
        return self.get_items.pop(0)

    def close(self) -> None:
        self.closed = True

    def join_thread(self) -> None:
        self.joined = True


class FakeProcess:
    pid = 12345

    def __init__(self) -> None:
        self.alive = True
        self.join_timeouts = []
        self.terminated = False
        self.killed = False
        self.closed = False

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout=None) -> None:
        self.join_timeouts.append(timeout)
        self.alive = False

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def close(self) -> None:
        self.closed = True


class FakeStartedExecutor(ProcessNodeExecutor):
    def __init__(self, result, *, idle_ready=None) -> None:
        super().__init__()
        self._result = result
        self._idle_ready = idle_ready
        self.put_items = []

    def _ensure_started(self) -> None:
        if self._process is not None and self._process.is_alive():
            return

        self._process = FakeProcess()
        self._job_queue = FakeQueue()
        get_items = []
        if self._idle_ready is not None:
            get_items.append(self._idle_ready)
        self._result_queue = FakeQueue(get_items=get_items)

    def _wait_for_result(self, node: NodeRef):
        assert self._job_queue is not None
        self.put_items = list(self._job_queue.items)
        return self._result


def _run_pids(trace_path: Path) -> dict[str, int]:
    rows = [
        line.split(',')
        for line in trace_path.read_text(encoding='utf-8').splitlines()
    ]
    return {
        node_id: int(pid)
        for event, node_id, pid in rows
        if event == 'run'
    }


def test_node_ref_does_not_use_cuda_by_default(tmp_path):
    with DAG('default_uses_cuda', out_dir=tmp_path):
        node = SourceNode(name='source', value=1)

    assert node.uses_cuda is False
    assert node.forces_cuda_cooldown is False


def test_forced_cooldown_restarts_and_runs_node_below_limit(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(process_executor_mod.time, 'sleep', sleeps.append)
    with DAG('forced_cooldown', out_dir=tmp_path):
        node = CudaCooldown(name='cooldown')
    output = {'image': 'image.png'}
    executor = FakeStartedExecutor(NodeResult(ok=True, output=output))
    executor._ensure_started()
    old_process = executor._process
    executor._cuda_count = 2
    executor._worker_used_cuda = True
    try:
        assert executor.run(
            node=node, out_dir=str(tmp_path), input_map={'default': output},
        ) == output
        assert old_process.closed
        assert executor._process is not old_process
        assert executor._cuda_count == 0
        assert sleeps == [executor.cuda_cooldown_after_restart]
        assert executor.put_items[-1].node is node
    finally:
        executor.close()


def test_forced_cooldown_without_worker_does_not_sleep(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(process_executor_mod.time, 'sleep', sleeps.append)
    with DAG('initial_cooldown', out_dir=tmp_path):
        node = CudaCooldown(name='cooldown')
    executor = FakeStartedExecutor(NodeResult(ok=True, output={}))
    try:
        assert executor.run(node=node, out_dir=str(tmp_path), input_map={}) == {}
        assert sleeps == []
    finally:
        executor.close()


def test_restarts_before_next_cuda_node_but_keeps_cpu_nodes_queued(tmp_path, monkeypatch):
    monkeypatch.setattr(process_executor_mod, '_CUDA_IDLE_MAX_WAIT', 0.1)
    trace_path = tmp_path / 'trace.csv'

    with DAG('process_limit', out_dir=tmp_path):
        cuda_1 = FileTracingNode(
            name='cuda_1',
            trace_path=str(trace_path),
            cuda=True,
        )
        cpu = FileTracingNode(
            name='cpu',
            trace_path=str(trace_path),
        )
        cuda_2 = FileTracingNode(
            name='cuda_2',
            trace_path=str(trace_path),
            cuda=True,
        )

    with ProcessNodeExecutor(
        max_cuda_nodes=1,
        cuda_shutdown_min_wait=0,
        cuda_cooldown_after_restart=0,
    ) as executor:
        executor.run(node=cuda_1, out_dir=str(tmp_path), input_map={})
        executor.run(node=cpu, out_dir=str(tmp_path), input_map={})
        executor.run(node=cuda_2, out_dir=str(tmp_path), input_map={})

    pids = _run_pids(trace_path)
    assert pids['cuda_1'] == pids['cpu']
    assert pids['cuda_2'] != pids['cuda_1']


def test_runs_post_run_in_worker_and_closes_after_failure(tmp_path):
    trace_path = tmp_path / 'trace.csv'

    with DAG('process_failure', out_dir=tmp_path):
        failing = FileTracingNode(
            name='failing',
            trace_path=str(trace_path),
            fail=True,
        )

    executor = ProcessNodeExecutor()
    with pytest.raises(ProcessNodeExecutionError, match='planned failure'):
        with executor:
            executor.run(
                node=failing,
                out_dir=str(tmp_path),
                input_map={},
            )

    assert executor._process is None
    events = trace_path.read_text(encoding='utf-8').splitlines()
    assert any(line.startswith('run,failing,') for line in events)
    assert not any(line.startswith('post_run,failing,') for line in events)


def test_keyboard_interrupt_waits_for_worker_teardown(tmp_path):
    trace_path = tmp_path / 'trace.csv'

    with DAG('process_interrupt', out_dir=tmp_path):
        node = SlowFileTracingNode(
            name='slow',
            trace_path=str(trace_path),
        )

    def interrupt_when_started() -> None:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if trace_path.exists():
                os.kill(os.getpid(), signal.SIGINT)
                time.sleep(0.1)
                os.kill(os.getpid(), signal.SIGINT)
                return
            time.sleep(0.01)

    interrupter = threading.Thread(target=interrupt_when_started)
    interrupter.start()

    executor = ProcessNodeExecutor()
    with pytest.raises(KeyboardInterrupt):
        with executor:
            executor.run(
                node=node,
                out_dir=str(tmp_path),
                input_map={},
            )

    interrupter.join(timeout=10)
    assert not interrupter.is_alive()
    assert executor._process is None

    events = trace_path.read_text(encoding='utf-8').splitlines()
    assert any(line.startswith('run,slow,') for line in events)
    assert any(line.startswith('post_run,slow,') for line in events)


def test_context_exit_uses_safe_shutdown_for_keyboard_interrupt(monkeypatch):
    executor = ProcessNodeExecutor()
    close_calls = []

    monkeypatch.setattr(
        executor,
        'close',
        lambda **kwargs: close_calls.append(kwargs),
    )

    executor.__exit__(KeyboardInterrupt, KeyboardInterrupt(), None)

    assert close_calls == [{
        'graceful_timeout': None,
        'terminate': False,
    }]


def test_close_skips_cuda_quiesce_for_cpu_only_worker():
    executor = ProcessNodeExecutor()
    process = FakeProcess()
    job_queue = FakeQueue()
    result_queue = FakeQueue()

    executor._process = process
    executor._job_queue = job_queue
    executor._result_queue = result_queue
    executor._worker_used_cuda = False

    executor.close(graceful_timeout=30.0)

    assert len(job_queue.items) == 1
    shutdown = job_queue.items[0]
    assert isinstance(shutdown, WorkerShutdown)
    assert shutdown.wait_for_cuda_idle is False
    assert process.join_timeouts == [30.0]
    assert process.terminated is False
    assert process.killed is False
    assert process.closed is True
    assert executor._process is None
    assert executor._worker_used_cuda is False


def test_close_extends_join_timeout_for_cuda_worker():
    executor = ProcessNodeExecutor()
    process = FakeProcess()
    job_queue = FakeQueue()
    result_queue = FakeQueue()

    executor._process = process
    executor._job_queue = job_queue
    executor._result_queue = result_queue
    executor._worker_used_cuda = True

    executor.close(graceful_timeout=30.0)

    assert len(job_queue.items) == 1
    shutdown = job_queue.items[0]
    assert isinstance(shutdown, WorkerShutdown)
    assert shutdown.wait_for_cuda_idle is True
    assert process.join_timeouts == [70.0]
    assert process.terminated is False
    assert process.killed is False
    assert process.closed is True
    assert executor._process is None
    assert executor._worker_used_cuda is False


def test_failed_cuda_node_marks_worker_as_cuda_used(tmp_path):
    with DAG('failed_cuda_marks_worker', out_dir=tmp_path):
        node = FileTracingNode(
            name='cuda_failing',
            trace_path=str(tmp_path / 'trace.csv'),
            cuda=True,
        )

    executor = FakeStartedExecutor(
        result=NodeResult(
            ok=False,
            error_type='RuntimeError',
            error_message='planned CUDA failure',
            traceback_text='',
        ),
        idle_ready=WorkerCudaIdleReady(),
    )

    with pytest.raises(ProcessNodeExecutionError, match='planned CUDA failure'):
        executor.run(
            node=node,
            out_dir=str(tmp_path),
            input_map={},
        )

    assert executor._worker_used_cuda is True
    assert executor._cuda_count == 0
    assert len(executor.put_items) == 2
    assert isinstance(executor.put_items[0], WorkerWaitForCudaIdle)


def test_cpu_node_does_not_request_pre_cuda_idle_wait(tmp_path):
    with DAG('cpu_node_no_pre_cuda_wait', out_dir=tmp_path):
        node = FileTracingNode(
            name='cpu',
            trace_path=str(tmp_path / 'trace.csv'),
            cuda=False,
        )

    executor = FakeStartedExecutor(
        result=NodeResult(ok=True, output={'ok': True})
    )

    executor.run(
        node=node,
        out_dir=str(tmp_path),
        input_map={},
    )

    assert len(executor.put_items) == 1
    assert not isinstance(executor.put_items[0], WorkerWaitForCudaIdle)
    assert executor._worker_cuda_idle_wait_done is False


def test_cuda_node_requests_pre_cuda_idle_wait_once(tmp_path):
    with DAG('cuda_node_pre_cuda_wait', out_dir=tmp_path):
        first = FileTracingNode(
            name='first',
            trace_path=str(tmp_path / 'trace.csv'),
            cuda=True,
        )
        second = FileTracingNode(
            name='second',
            trace_path=str(tmp_path / 'trace.csv'),
            cuda=True,
        )

    executor = FakeStartedExecutor(
        result=NodeResult(ok=True, output={'ok': True}),
        idle_ready=WorkerCudaIdleReady(),
    )

    executor.run(
        node=first,
        out_dir=str(tmp_path),
        input_map={},
    )
    first_puts = list(executor.put_items)

    executor._result = NodeResult(ok=True, output={'ok': True})
    executor.run(
        node=second,
        out_dir=str(tmp_path),
        input_map={},
    )
    second_puts = list(executor.put_items)

    assert isinstance(first_puts[0], WorkerWaitForCudaIdle)
    assert len(first_puts) == 2
    assert not isinstance(first_puts[1], WorkerWaitForCudaIdle)
    assert len(second_puts) == 3
    assert sum(isinstance(item, WorkerWaitForCudaIdle)
               for item in second_puts) == 1
    assert executor._worker_cuda_idle_wait_done is True


def test_pre_cuda_idle_timeout_baseline_does_not_configure_thresholds(tmp_path):
    with DAG('cuda_idle_timeout_baseline', out_dir=tmp_path):
        node = FileTracingNode(
            name='cuda',
            trace_path=str(tmp_path / 'trace.csv'),
            cuda=True,
        )

    timeout_baseline = process_executor_mod.NvidiaIdleBaseline(
        temperature_celsius=35.0,
        power_watts=40.0,
        pstate=5,
        gpu_utilization_percent=0,
        memory_used_mib=0,
    )
    executor = FakeStartedExecutor(
        result=NodeResult(ok=True, output={'ok': True}),
        idle_ready=WorkerCudaIdleReady(
            idle_reached=False,
            baseline=timeout_baseline,
            error_message='NVML pre-CUDA idle wait timed out',
        ),
    )

    executor.run(
        node=node,
        out_dir=str(tmp_path),
        input_map={},
    )

    assert executor._worker_cuda_idle_wait_done is True
    assert executor._cuda_idle_baseline is None


def test_pre_cuda_idle_reached_configures_thresholds(tmp_path):
    with DAG('cuda_idle_reached_baseline', out_dir=tmp_path):
        node = FileTracingNode(
            name='cuda',
            trace_path=str(tmp_path / 'trace.csv'),
            cuda=True,
        )

    idle_baseline = process_executor_mod.NvidiaIdleBaseline(
        temperature_celsius=35.0,
        power_watts=40.0,
        pstate=5,
        gpu_utilization_percent=0,
        memory_used_mib=0,
    )
    executor = FakeStartedExecutor(
        result=NodeResult(ok=True, output={'ok': True}),
        idle_ready=WorkerCudaIdleReady(
            idle_reached=True,
            baseline=idle_baseline,
        ),
    )

    executor.run(
        node=node,
        out_dir=str(tmp_path),
        input_map={},
    )

    assert executor._worker_cuda_idle_wait_done is True
    assert executor._cuda_idle_baseline == idle_baseline


def test_dag_runner_dispatches_nodes_to_process_executor(tmp_path):
    trace_path = tmp_path / 'trace.csv'

    with DAG('process_dag_runner', out_dir=tmp_path) as dag:
        first = FileTracingNode(
            name='first',
            trace_path=str(trace_path),
        )
        second = FileTracingNode(
            name='second',
            trace_path=str(trace_path),
        )
        first >> second

    DAGRunner(dag).run()

    pids = _run_pids(trace_path)
    assert pids['first'] == pids['second']
    assert pids['first'] != os.getpid()

    events = trace_path.read_text(encoding='utf-8').splitlines()
    assert sum(line.startswith('post_run,') for line in events) == 2
