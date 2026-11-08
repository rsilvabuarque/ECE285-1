"""Integrity, relocation and numerical replay guards using synthetic packets."""
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
import zipfile

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/"scripts/conditioning_20261107"
sys.path.insert(0,str(SCRIPTS))
import replay_extensions as replay
import package_extensions as package


def archive(path, members, *, manifest_name="manifest_ext.json", duplicate=None, symlink=None):
    manifest = dict(files={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in members.items()})
    with zipfile.ZipFile(path,"w") as z:
        for name,data in members.items():z.writestr(name,data)
        z.writestr(manifest_name,json.dumps(manifest))
        if duplicate:z.writestr(duplicate,members[duplicate])
        if symlink:
            info=zipfile.ZipInfo(symlink);info.external_attr=(stat.S_IFLNK|0o777)<<16;z.writestr(info,"target")
    return replay.digest(path)


def test_archive_binds_outer_and_every_member(tmp_path):
    path=tmp_path/"payload.zip";sha=archive(path,{"original/protocol.json":b'{"energy":1.25}',"scores.csv":b"id,E\na,2.0\n"})
    m=replay.verify_archive(path,sha,"manifest_ext.json")
    assert len(m["files"])==2
    with pytest.raises(ValueError,match="archive SHA256"):
        replay.verify_archive(path,"0"*64,"manifest_ext.json")
    with zipfile.ZipFile(path,"a") as z:z.writestr("unexpected.txt","not manifested")
    with pytest.raises(ValueError,match="members differ"):
        replay.verify_archive(path,replay.digest(path),"manifest_ext.json")


@pytest.mark.parametrize("bad",["../escape","/absolute","C:/drive","back\\slash"])
def test_zip_path_traversal_rejected(tmp_path,bad):
    path=tmp_path/"bad.zip";sha=archive(path,{bad:b"x"})
    with pytest.raises(ValueError,match="Unsafe"):
        replay.verify_archive(path,sha,"manifest_ext.json")


def test_duplicate_symlink_and_wrong_member_hash_rejected(tmp_path):
    duplicate=tmp_path/"duplicate.zip"
    with pytest.warns(UserWarning,match="Duplicate"):
        sha=archive(duplicate,{"a":b"x"},duplicate="a")
    with pytest.raises(ValueError,match="Duplicate"):
        replay.verify_archive(duplicate,sha,"manifest_ext.json")
    link=tmp_path/"link.zip";sha=archive(link,{"a":b"x"},symlink="link")
    with pytest.raises(ValueError,match="symlinks"):
        replay.verify_archive(link,sha,"manifest_ext.json")
    wrong=tmp_path/"wrong.zip"
    with zipfile.ZipFile(wrong,"w") as z:
        z.writestr("a",b"x");z.writestr("manifest_ext.json",json.dumps(dict(files={"a":dict(bytes=1,sha256="0"*64)})))
    with pytest.raises(ValueError,match="Member SHA256"):
        replay.verify_archive(wrong,replay.digest(wrong),"manifest_ext.json")


def test_path_only_view_preserves_protocol_bytes_hash_and_science(tmp_path):
    root=Path("/historic/project");mirror=tmp_path/"new/project";mirror.mkdir(parents=True)
    original={"sources":{"data":{"path":str(root/"data.npz"),"sha256":"abc"}},"beta":.1,"energy":-1.25,"conditions":[12,"low"]}
    path=tmp_path/"protocol.json";path.write_text(json.dumps(original));before=replay.digest(path)
    relocate=replay.Relocator(root,mirror,{"data.npz":{}})
    derived=relocate.json_proxy().loads(path.read_text())
    assert derived["sources"]["data"]["path"]==str(mirror/"data.npz")
    assert derived["sources"]["data"]["sha256"]=="abc"
    assert {k:derived[k] for k in ("beta","energy","conditions")}=={k:original[k] for k in ("beta","energy","conditions")}
    assert replay.digest(path)==before
    with pytest.raises(ValueError,match="Unbound historical"):
        relocate.path(root/"private/not_in_packet.npz")


def test_csv_identity_counts_order_missing_and_roundoff(tmp_path):
    actual=tmp_path/"actual.csv";expected=tmp_path/"expected.csv"
    expected.write_text("id,count,E\na,2,1.0000000000000000\nb,1,-3.0\n")
    actual.write_text("id,count,E\na,2,1.0000000000000002\nb,1,-3.0\n")
    assert replay.compare_csv(actual,expected)["fields_checked"]==9
    for text in ("id,count,E\na,3,1.0\nb,1,-3.0\n", "id,count,E\nb,1,-3.0\na,2,1.0\n",
                 "id,count,E\na,2,1.00000000001\nb,1,-3.0\n", "id,count,E\na,2,\nb,1,-3.0\n"):
        actual.write_text(text)
        with pytest.raises(ValueError):replay.compare_csv(actual,expected)


def test_materialization_authenticates_locator_and_preserves_originals(tmp_path):
    core=tmp_path/"core.zip";ext=tmp_path/"ext.zip";data=b"immutable original"
    archive(core,{"data/a":data},manifest_name="manifest.json")
    archive(ext,{"project/new.txt":b"new","raw_shards_inventory.json":b'{"raw_shards":[]}'})
    binding=lambda packet,member,data:dict(packet=packet,member=member,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    config=dict(locator={"old/data.txt":binding("core","data/a",data),"new.txt":binding("extensions","project/new.txt",b"new")})
    mirror=replay.materialize(core,ext,config,tmp_path/"relocated/project")
    assert (mirror/"old/data.txt").read_bytes()==data
    config["locator"]["new.txt"]["sha256"]="0"*64
    with pytest.raises(ValueError,match="Materialized source"):
        replay.materialize(core,ext,config,tmp_path/"broken/project")


def test_curator_allows_only_new_editor_weights_and_reuses_core_hash(tmp_path):
    root=tmp_path/"project";root.mkdir();source=root/"data.npz";source.write_bytes(b"same core bytes")
    sha=replay.digest(source)
    c=package.ExtensionCurator(root,dict(files={"data/core.npz":dict(bytes=source.stat().st_size,sha256=sha)},source_provenance={}))
    c.add(source);assert c.locator["data.npz"]["packet"]=="core";assert not c.files
    for relative in ("for_chat_gpt/private.pdf","models/foundation.pt","runs/conditioning_20261107/fits/other/best.pt","runs/conditioning_20261107/oracle/scores/uma/raw.npz"):
        p=root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b"excluded")
        with pytest.raises(ValueError):c.add(p)
    good=root/"runs/conditioning_20261107/fits/editor_20261106/best.pt";good.parent.mkdir(parents=True);good.write_bytes(b"editor")
    c.add(good);assert c.locator[good.relative_to(root).as_posix()]["packet"]=="extensions"


def test_training_command_relocates_all_paths_without_changing_fit_parameters(tmp_path):
    original="/old/project";root=tmp_path/"verified";tasks=root/"tasks.json";root.mkdir()
    task=dict(id="editor_20261106",trainer_module="cathode_cvae.learning.train_edit",argv=["--root",original,"--data",original+"/data.npz","--seed","20261106","--max-updates","4000","--device","cuda","--output-dir",original+"/fits/editor_20261106"])
    tasks.write_text(json.dumps(dict(tasks=[task])))
    cfg=dict(original_root=original,training_tasks="tasks.json",locator={"data.npz":{}})
    command=replay.training_command(cfg,root,"editor_20261106",tmp_path/"fresh_fit","cpu")
    assert command[command.index("--data")+1]==str(root/"data.npz")
    assert command[command.index("--root")+1]==str(root)
    assert command[command.index("--max-updates")+1]=="4000"
    assert command[command.index("--device")+1]=="cpu"
    assert "/old/project" not in " ".join(command)
    assert json.loads(tasks.read_text())["tasks"][0]==task


def test_verified_script_adapter_loads_relocated_arrays_and_refuses_live_fallback(tmp_path):
    import numpy as np
    original=tmp_path/"historic";original.mkdir();mirror=tmp_path/"fresh/verified_project";mirror.mkdir(parents=True)
    np.savez(original/"not_released.npz",value=np.array([99]))
    np.savez(mirror/"array.npz",value=np.array([7]))
    protocol=mirror/"protocol.json";protocol.write_text(json.dumps(dict(sources=dict(data=dict(path=str(original/"array.npz"),sha256="preserved")),energy=-2.5)))
    scripts=mirror/"scripts/conditioning_20261107";scripts.mkdir(parents=True)
    (scripts/"synthetic.py").write_text("import json\nfrom pathlib import Path\nimport numpy as np\ndef read(p):\n c=json.loads(Path(p).read_text())\n with np.load(c['sources']['data']['path']) as z:return int(z['value'][0]),c['energy'],c['sources']['data']['sha256']\n")
    relocate=replay.Relocator(original,mirror,{"array.npz":{}});before=replay.digest(protocol)
    module=replay.load_script(mirror,"synthetic",relocate)
    assert module.read(protocol)==(7,-2.5,"preserved")
    assert replay.digest(protocol)==before
    with pytest.raises(ValueError,match="Unbound historical"):
        module.np.load(original/"not_released.npz")


def test_relocation_allows_new_outputs_beneath_fresh_output_root(tmp_path):
    original=tmp_path/"original";output=original/"new_replay";mirror=output/"verified_project"
    relocate=replay.Relocator(original,mirror,{})
    assert relocate.path(output/"editor/receipt.json")==output/"editor/receipt.json"
    with pytest.raises(ValueError,match="Unbound historical"):
        relocate.path(original/"old_unverified_receipt.json")


def test_mutable_historical_path_resolves_version_by_hash_not_filename(tmp_path):
    old="/historic/project";mirror=tmp_path/"verified_project"
    early="1"*64;later="2"*64
    relocate=replay.Relocator(old,mirror,{"oracle/registry.json":dict(sha256=early),"snapshots/pre_aggregate.json":dict(sha256=later)})
    original=dict(path=old+"/oracle/registry.json",sha256=later,scientific_parameter=.1)
    view=relocate.replace(original)
    assert view["path"]==str(mirror/"snapshots/pre_aggregate.json")
    assert view["sha256"]==later and original["path"]==old+"/oracle/registry.json"
    assert relocate.versioned_overrides[(original["path"],later)]==view["path"]
    missing=dict(path=original["path"],sha256="3"*64)
    assert relocate.replace(missing)==missing
    assert relocate.unresolved_metadata[(original["path"],"3"*64)]["scope"].startswith("Unresolved historical metadata")
    with pytest.raises(ValueError,match="Unresolved historical metadata cannot be opened"):
        relocate.path(original["path"])


def test_curator_keeps_previous_verified_version_when_explicit_source_updates(tmp_path):
    root=tmp_path/"project";root.mkdir();source=root/"report.json";source.write_bytes(b"old")
    old=replay.digest(source)
    core=dict(files={"report.json":dict(bytes=3,sha256=old)},source_provenance={"report.json":"report.json"})
    curator=package.ExtensionCurator(root,core);source.write_bytes(b"new report")
    curator.add(source)
    preserved=[v for v in curator.locator.values() if v["sha256"]==old]
    assert len(preserved)==1 and preserved[0]["packet"]=="core"
    assert curator.locator["report.json"]["packet"]=="extensions"
    relocate=replay.Relocator(root,tmp_path/"mirror",curator.locator)
    assert ".historical_versions" in relocate.replace(dict(path=str(source),sha256=old))["path"]


def test_unresolved_old_version_blocks_path_array_and_hash_despite_newer_mapping(tmp_path):
    import numpy as np
    old=tmp_path/"historical";old.mkdir();mirror=tmp_path/"fresh/verified_project";mirror.mkdir(parents=True)
    np.savez(old/"registry.npz",value=[99]);np.savez(mirror/"registry.npz",value=[88])
    scripts=mirror/"scripts/conditioning_20261107";scripts.mkdir(parents=True)
    (scripts/"synthetic_hash.py").write_text("from pathlib import Path\nimport numpy as np\ndef sha256(p):return 'unpatched'\n")
    relocate=replay.Relocator(old,mirror,{"registry.npz":dict(sha256="1"*64)})
    metadata=dict(path=str(old/"registry.npz"),sha256_at_freeze="2"*64,witness="immutable registered_history.json")
    assert relocate.replace(metadata)==metadata
    module=replay.load_script(mirror,"synthetic_hash",relocate)
    for access in (module.Path,module.np.load,module.sha256):
        with pytest.raises(ValueError,match="Unresolved historical metadata cannot be opened"):
            access(old/"registry.npz")
    # A separately hash-bound valid source still resolves explicitly to its
    # authenticated supplied version, rather than opening the old filename.
    assert relocate.replace(dict(path=str(old/"registry.npz"),sha256="1"*64))["path"]==str(mirror/"registry.npz")


def test_versioned_download_extraction_path_stays_inside_project(tmp_path):
    root=tmp_path/"project"
    assert package.extract_destination(root,root/"artifacts/conditioning_20261107/extensions_v2")=="artifacts/conditioning_20261107/extensions_v2"
    with pytest.raises(ValueError):package.extract_destination(root,tmp_path/"outside")
    with pytest.raises(ValueError):package.extract_destination(root,root)
