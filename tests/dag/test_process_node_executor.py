from __future__ import annotations

import os
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pytest

from morphalo.dag import DAG, NodeRef
from morphalo.dag.process_executor import (
    ProcessNodeExecutionError,
    ProcessNodeExecutor,
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


def test_restarts_before_next_cuda_node_but_keeps_cpu_nodes_queued(tmp_path):
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

    with ProcessNodeExecutor(max_cuda_nodes=1) as executor:
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
