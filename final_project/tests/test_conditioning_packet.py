"""Protect portable replay identities and write-once packet curation."""
import importlib.util
import json
from pathlib import Path

import pytest


def module(name):
    path = Path(__file__).resolve().parents[1] / "scripts/conditioning_20261107" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


package = module("package_results")
replay = module("replay_cached_results")


def test_curator_excludes_private_documents_and_external_weights(tmp_path):
    private = tmp_path / "for_chat_gpt/input.pdf"
    private.parent.mkdir(); private.write_bytes(b"private")
    curator = package.Curator(tmp_path, tmp_path / "stage")
    with pytest.raises(ValueError, match="Private/weight"):
        curator.add(private, "input.pdf")
    weight = tmp_path / "foundation.pt"; weight.write_bytes(b"weight")
    with pytest.raises(ValueError, match="Only the required"):
        curator.add(weight, "weight.pt")
    public = tmp_path / "report.pdf"; public.write_bytes(b"%PDF")
    with pytest.raises(ValueError, match="Only generated"):
        curator.add(public, "report.pdf")
    with pytest.raises(ValueError, match="Unsafe"):
        package.safe_relative("../escape")


def test_curator_conflict_preflight_preserves_every_destination(tmp_path):
    stage, target = tmp_path / "stage", tmp_path / "target"
    stage.mkdir(); target.mkdir()
    (stage / "a_new.txt").write_text("new")
    (stage / "z_conflict.json").write_text("new metadata")
    (target / "z_conflict.json").write_text("preserve this")
    with pytest.raises(FileExistsError): package.preserve_existing(stage, target)
    assert not (target / "a_new.txt").exists()
    assert (target / "z_conflict.json").read_text() == "preserve this"


def test_training_paths_are_portable_without_changing_optimizer_budget(tmp_path):
    root = tmp_path
    argv = ["--data", str(root / "data.npz"), "--parent", str(root / "p.npz"),
        "--condition-schema-in", str(root / "schema.json"), "--max-updates", "4000",
        "--device", "cuda", "--output-dir", str(root / "fits/graph_context_20261106")]
    result = package.portable_argv(argv, root)
    assert result[result.index("--max-updates") + 1] == "4000"
    assert "{packet}/data/training_round0.npz" in result
    assert "{output}/graph_context_20261106" in result
    assert str(root) not in " ".join(result)
    bad = argv + ["--unexpected", str(root / "unresolved")]
    with pytest.raises(ValueError, match="unresolved"): package.portable_argv(bad, root)


def test_csv_replay_preserves_counts_ids_rows_and_missing_values(tmp_path):
    ref, out = tmp_path / "ref.csv", tmp_path / "out.csv"
    ref.write_text("id,count,energy\nmask48_abc,128,-5.1234567890000\n")
    out.write_text("id,count,energy\nmask48_abc,128,-5.1234567890001\n")
    assert replay.compare_csv(out, ref)["rows_including_header"] == 2
    out.write_text("id,count,energy\nmask48_abc,127,-5.1234567890000\n")
    with pytest.raises(ValueError, match="identity/count"): replay.compare_csv(out, ref)
    out.write_text("id,count,energy\nmask48_def,128,-5.1234567890000\n")
    with pytest.raises(ValueError): replay.compare_csv(out, ref)
    out.write_text("id,count,energy\nmask48_abc,128,\n")
    with pytest.raises(ValueError): replay.compare_csv(out, ref)
    out.write_text(ref.read_text() + "mask48_other,128,-5.1\n")
    with pytest.raises(ValueError, match="shape"): replay.compare_csv(out, ref)


def test_replay_manifest_detects_modified_payload_and_sources(tmp_path):
    packet, source = tmp_path / "packet", tmp_path / "repo"
    packet.mkdir(); (source / "src").mkdir(parents=True)
    (source / "src/example.py").write_text("# exact evaluator\n")
    hashes = {"src/example.py": replay.digest(source / "src/example.py")}
    (packet / "replay.json").write_text(json.dumps(dict(source_hashes=hashes)))
    manifest = dict(release_tag="ece285-conditioning-20261108", core_full_fit_count=28, source_hashes=hashes,
        files={"replay.json": dict(bytes=(packet / "replay.json").stat().st_size, sha256=replay.digest(packet / "replay.json"))})
    (packet / "manifest.json").write_text(json.dumps(manifest))
    assert replay.verify_packet(packet, source)[0]["core_full_fit_count"] == 28
    (source / "src/example.py").write_text("# changed evaluator\n")
    with pytest.raises(ValueError, match="Runtime source"): replay.verify_packet(packet, source)
    (packet / "replay.json").write_text("{}")
    with pytest.raises(ValueError, match="Payload binding"): replay.verify_packet(packet, source)


def test_required_readiness_refuses_incomplete_or_changed_evidence(tmp_path):
    directories = [tmp_path / "final", tmp_path / "matched"]
    for directory, count, teachers in zip(directories, [34560, 20736], [["uma"], ["uma", "mace"]]):
        directory.mkdir()
        (directory / "receipt.json").write_text(json.dumps(dict(status="PASS", attempt_rows=count,
            teachers=teachers, canonical_keys_recomputed=True)))
    cost, pdf = tmp_path / "cost.json", tmp_path / "main.pdf"
    cost.write_text(json.dumps(dict(findings=[], evidence_stable_during_audit=True,
        training=dict(eligible_matched_round0_4000_update_trajectories=28))))
    pdf.write_bytes(b"%PDF-frozen")
    assert package.check_ready(*directories, cost, pdf)["evidence_stable_during_audit"]
    value = json.loads((directories[0] / "receipt.json").read_text()); value["attempt_rows"] -= 1
    (directories[0] / "receipt.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="not complete"): package.check_ready(*directories, cost, pdf)


def test_final_evaluation_pass_receipt_must_bind_actual_inputs(tmp_path):
    source = tmp_path / "scores.npz"; source.write_bytes(b"exact saved scalar scores")
    (tmp_path / "receipt.json").write_text(json.dumps(dict(status="PASS",
        source_hashes=dict(scores_uma=package.digest(source)))))
    package.check_evaluation_bindings(tmp_path, dict(scores_uma=source))
    source.write_bytes(b"different scores with same filename")
    with pytest.raises(ValueError, match="input changed"):
        package.check_evaluation_bindings(tmp_path, dict(scores_uma=source))
