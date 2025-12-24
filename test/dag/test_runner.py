from test.dag.nodes import *

from stability.dag import DAG
from stability.dag.runner import DAGRunner


def test_linear_dag():
    with DAG('linear', out_dir='/tmp') as dag:
        a = SourceNode(id='A', value=10)
        b = PassNode(id='B')

        a >> b

    runner = DAGRunner(dag)
    runner.run()


def test_merge_dag():
    with DAG('merge', out_dir='/tmp') as dag:
        a = SourceNode(id='A', value=1)
        x = PassNode(id='X')

        a >> x 
        b = SourceNode(id='B', value=2)
        c = MergeNode(id='C')

        b >> c
        x >> c.getAttachmentSink(id='X', input_id='attachment')

    runner = DAGRunner(dag)
    runner.run()


def test_loop():
    with DAG('deadlock', out_dir='/tmp') as dag:
        a = PassNode(id='A')
        b = PassNode(id='B')

        a >> b
        b >> a   # loop

    runner = DAGRunner(dag)
    try:
        runner.run()
    except RuntimeError as e:
        print('OK deadlock detected:', e)


if __name__ == '__main__':
    # test_linear_dag()
    test_merge_dag()
    # test_loop()
