from pathlib import Path

import pytest

from morphalo.dag import DAG
from morphalo.nodes import FileImage


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('not necessarily an image')
    return path


def test_file_image_exposes_single_file_as_image(tmp_path):
    src = _touch(tmp_path / 'one.txt')

    with DAG('file_image_single', tmp_path):
        node = FileImage(name='src', path=src)

    out = node.run(tmp_path)

    assert out['image'] == str(src.resolve())
    assert 'images' not in out
    assert out['sources'] == [
        {
            'source': str(src.resolve()),
            'type': 'file',
            'files': [str(src.resolve())],
        }
    ]


def test_file_image_preserves_explicit_file_list_order(tmp_path):
    a = _touch(tmp_path / 'a.txt')
    b = _touch(tmp_path / 'b.txt')

    with DAG('file_image_list', tmp_path):
        node = FileImage(name='src', path=[b, a])

    out = node.run(tmp_path)

    assert out['images'] == [str(b.resolve()), str(a.resolve())]
    assert 'image' not in out
    assert [s['source'] for s in out['sources']] == [
        str(b.resolve()),
        str(a.resolve()),
    ]


def test_file_image_accepts_non_list_sequences(tmp_path):
    a = _touch(tmp_path / 'a.txt')
    b = _touch(tmp_path / 'b.txt')

    with DAG('file_image_tuple', tmp_path):
        node = FileImage(name='src', path=(b, a))

    out = node.run(tmp_path)

    assert out['images'] == [str(b.resolve()), str(a.resolve())]


def test_file_image_expands_direct_directory_files_non_recursively(tmp_path):
    root = tmp_path / 'refs'
    first = _touch(root / 'b.any')
    second = _touch(root / 'a.any')
    _touch(root / 'nested' / 'ignored.any')

    with DAG('file_image_dir', tmp_path):
        node = FileImage(name='src', path=root)

    out = node.run(tmp_path)

    assert out['images'] == [
        str(second.resolve()),
        str(first.resolve()),
    ]
    assert out['sources'] == [
        {
            'source': str(root.resolve()),
            'type': 'directory',
            'recursive': False,
            'files': [
                str(second.resolve()),
                str(first.resolve()),
            ],
        }
    ]


def test_file_image_expands_mixed_files_and_directories_in_input_order(tmp_path):
    before = _touch(tmp_path / 'before.dat')
    after = _touch(tmp_path / 'after.dat')
    refs = tmp_path / 'refs'
    ref_a = _touch(refs / 'a.dat')
    ref_b = _touch(refs / 'b.dat')

    with DAG('file_image_mixed', tmp_path):
        node = FileImage(name='src', path=[before, refs, after])

    out = node.run(tmp_path)

    assert out['images'] == [
        str(before.resolve()),
        str(ref_a.resolve()),
        str(ref_b.resolve()),
        str(after.resolve()),
    ]


def test_file_image_ignores_empty_directories_but_fails_if_no_files(tmp_path):
    empty = tmp_path / 'empty'
    empty.mkdir()

    with DAG('file_image_empty', tmp_path):
        node = FileImage(name='src', path=empty)

    with pytest.raises(ValueError, match='no files found'):
        node.run(tmp_path)
