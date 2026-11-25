import check_authors
import check_commit_msg
import check_files


def test_blocks_media_and_data() -> None:
    assert check_files.check_path("data/x.txt", 1, 1000)
    assert check_files.check_path("clips/a.MP4", 1, 1000)
    assert check_files.check_path("m/model.onnx", 1, 1000)
    assert check_files.check_path(".env", 1, 1000)
    assert check_files.check_path("a.py", 5000, 1000)


def test_allows_normal_files() -> None:
    assert not check_files.check_path("hello_vision/a.py", 10, 1000)
    assert not check_files.check_path(".env.example", 10, 1000)
    assert not check_files.check_path("docs/diagram.svg", 10, 1000)


def test_commit_message_trailers() -> None:
    assert check_commit_msg.find_violations("fix\n\nCo-Authored-By: X <x@y>")
    assert check_commit_msg.find_violations("Generated with a tool")
    assert not check_commit_msg.find_violations("Add catalog loader")


def test_author_audit() -> None:
    assert not check_authors.audit(["A <a@x>", "A <a@x>"], "msg")
    assert check_authors.audit(["A <a@x>", "B <b@x>"], "msg")
    assert check_authors.audit(["A <a@x>"], "Co-authored-by: B")
