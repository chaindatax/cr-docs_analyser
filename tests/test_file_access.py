


# --- first_file_per_leaf_subdir ---

def test_returns_one_file_per_leaf_subdir(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "f2.jpg").write_bytes(b"")
    (tmp_path / "a" / "f1.jpg").write_bytes(b"")
    (tmp_path / "b" / "sub").mkdir(parents=True)
    (tmp_path / "b" / "sub" / "f3.jpg").write_bytes(b"")
    (tmp_path / "b" / "sub" / "f1.jpg").write_bytes(b"")

    from tests.test_dataset import first_file_per_leaf_subdir
    result = first_file_per_leaf_subdir(tmp_path)

    assert len(result) == 2
    assert result[0] == tmp_path / "a" / "f1.jpg"
    assert result[1] == tmp_path / "b" / "sub" / "f1.jpg"


def test_excludes_non_leaf_dirs(tmp_path):
    (tmp_path / "parent" / "child").mkdir(parents=True)
    (tmp_path / "parent" / "child" / "doc.jpg").write_bytes(b"")
    (tmp_path / "parent" / "sibling").mkdir()
    (tmp_path / "parent" / "sibling" / "doc.jpg").write_bytes(b"")

    from tests.test_dataset import first_file_per_leaf_subdir
    result = first_file_per_leaf_subdir(tmp_path)

    dirs = {p.parent.name for p in result}
    assert "parent" not in dirs
    assert dirs == {"child", "sibling"}


def test_empty_root_returns_empty_list(tmp_path):
    from tests.test_dataset import first_file_per_leaf_subdir
    assert first_file_per_leaf_subdir(tmp_path) == []

