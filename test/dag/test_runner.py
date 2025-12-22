from test.dag.nodes import *

from stability.dag import DAG
from stability.dag.runner import DAGRunner


def test_linear_dag():
    with DAG('linear', out_dir='/tmp') as dag:
        a = SourceNode(op='source', id='A', value=10)
        b = PassNode(op='pass', id='B')

        a >> b

    runner = DAGRunner(dag)
    runner.run()


def test_merge_dag():
    with DAG('merge', out_dir='/tmp') as dag:
        a = SourceNode(op='source', id='A', value=1)
        b = SourceNode(op='source', id='B', value=2)
        c = MergeNode(op='merge', id='C')

        a >> c
        b >> c.getAttachmentSink(id='X', input_id='attachment')

    runner = DAGRunner(dag)
    runner.run()


def test_loop():
    with DAG('deadlock', out_dir='/tmp') as dag:
        a = PassNode(op='pass', id='A')
        b = PassNode(op='pass', id='B')

        a >> b
        b >> a   # loop

    runner = DAGRunner(dag)
    try:
        runner.run()
    except RuntimeError as e:
        print('OK deadlock detected:', e)


if __name__ == '__main__':
    test_linear_dag()
    test_merge_dag()
    test_loop()
