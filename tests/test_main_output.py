"""main.py end-to-end contract with a fake model: manifest -> exact CSV, standard quoting, quiet logs."""
import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_main(monkeypatch, data_dir, out_path):
    monkeypatch.setenv("LIT_DATA_DIR", str(data_dir))
    monkeypatch.setenv("LIT_SUBMISSION_PATH", str(out_path))
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    monkeypatch.syspath_prepend(str(ROOT / "submission_src"))
    spec = importlib.util.spec_from_file_location("lit_main", ROOT / "submission_src" / "main.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_metadata_manifest_quoting_and_quiet_logs(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    (data / "clips").mkdir(parents=True)
    names = ["aaaa1111.mp3", "bbbb2222.mp3", "cccc3333.mp3"]
    with open(data / "test_metadata.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audio_filename", "file_duration_seconds", "language"])
        for n in names:
            w.writerow([n, 5, "enspa"])
    out = tmp_path / "submission" / "submission.csv"
    m = _load_main(monkeypatch, data, out)

    class FakeT:
        device = "cpu"

    monkeypatch.setattr(m, "Transcriber", lambda *a, **k: FakeT())
    fake = ['plain text', 'has, a comma and "quotes"', 'multi\nline  text']
    monkeypatch.setattr(m, "transcribe_many", lambda t, paths: fake)
    m.main()

    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["audio_filename", "transcript"]  # exactly two columns, no metadata columns
    assert [r[0] for r in rows[1:]] == names
    assert rows[2][1] == 'has, a comma and "quotes"'  # survives standard CSV quoting
    assert rows[3][1] == "multi line text"  # newlines never reach the file
    printed = capsys.readouterr().out
    assert not any(n[:-4] in printed for n in names)  # no clip names in logs


def test_falls_back_to_submission_format_and_listing(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "clips").mkdir(parents=True)
    (data / "clips" / "x.mp3").write_bytes(b"0")
    m = _load_main(monkeypatch, data, tmp_path / "o.csv")
    assert m.read_clip_names() == ["x.mp3"]
    (data / "submission_format.csv").write_text("audio_filename,transcript\ny.mp3,hello world\n")
    assert m.read_clip_names() == ["y.mp3"]


def test_empty_and_na_like_transcripts_never_become_nan(tmp_path, monkeypatch):
    """Regression for the rejected submission: an empty transcript is read back as NaN by pandas."""
    import pandas as pd

    from lit.normalize import norm

    data = tmp_path / "data"
    (data / "clips").mkdir(parents=True)
    names = [f"c{i}.mp3" for i in range(8)]
    with open(data / "test_metadata.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audio_filename", "file_duration_seconds", "language"])
        for n in names:
            w.writerow([n, 3, "enspa"])
    out = tmp_path / "submission" / "submission.csv"
    m = _load_main(monkeypatch, data, out)

    class FakeT:
        device = "cpu"

    monkeypatch.setattr(m, "Transcriber", lambda *a, **k: FakeT())
    fake = ["", "   ", "NA", "None", "null", "hola, que tal", None, "a b\x00c"]
    monkeypatch.setattr(m, "transcribe_many", lambda t, paths: fake)
    m.main()

    df = pd.read_csv(out)  # pandas defaults, exactly how a validator would read it
    assert list(df.columns) == ["audio_filename", "transcript"]
    assert df["audio_filename"].tolist() == names
    assert not df["transcript"].isna().any()
    assert df["transcript"].map(lambda s: isinstance(s, str) and s.strip() != "").all()
    assert norm(df["transcript"][2]) == norm("NA") and norm(df["transcript"][3]) == norm("None")  # scored text unchanged
    assert df["transcript"][5] == "hola, que tal"
    assert "\n" not in "".join(df["transcript"]) and " " not in "".join(df["transcript"])
