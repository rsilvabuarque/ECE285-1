"""Allocated CPU verification launcher; no source checkout is a replay input.

Copy the two artifacts to a new user-specified directory, authenticate outer
archives and every member, then extract the bootstrap from that authenticated
extension. Its subprocess restores all scientific inputs/source from the ZIPs.
This launcher uses only stdlib; the cached evaluator needs ordinary project
dependencies, with no backend SDK or foundation weights.
"""
import argparse
from datetime import datetime,timezone
import hashlib,importlib.metadata,json,os,shutil,subprocess,sys,time,traceback,zipfile
from pathlib import Path

CORE_SHA="b720c932edf7f48cf1d8789f39abe9135ca9f9b17a5322b4558545d3243a72df"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(2**20),b""):h.update(block)
    return h.hexdigest()


def verify(path,outer,manifest_name):
    if sha(path)!=outer:raise ValueError("Outer archive SHA differs")
    with zipfile.ZipFile(path) as z:
        manifest=json.loads(z.read(manifest_name));names=z.namelist()
        if len(names)!=len(set(names)) or set(names)!=set(manifest["files"])|{manifest_name}:raise ValueError("Archive member list differs")
        for name,binding in manifest["files"].items():
            h=hashlib.sha256();count=0
            with z.open(name) as f:
                for block in iter(lambda:f.read(2**20),b""):h.update(block);count+=len(block)
            if count!=binding["bytes"] or h.hexdigest()!=binding["sha256"]:raise ValueError("Archive member differs: "+name)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True)
    p.add_argument("--fresh-dir",type=Path,required=True);p.add_argument("--receipt-dir",type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();fresh=a.fresh_dir.resolve();receipts=a.receipt_dir.resolve()
    fresh.mkdir(exist_ok=False);receipts.mkdir(parents=True,exist_ok=True)
    started=datetime.now(timezone.utc).isoformat();clock=time.perf_counter()
    result=dict(status="STARTED",allocation=os.environ.get("SLURM_JOB_ID"),python=sys.executable,started_UTC=started,
        command=[sys.executable,*sys.argv],PYTHONPATH_unset_at_bootstrap=True,new_MLIP_calls=0,new_training_updates=0)
    code=1
    try:
        paths={"core.zip":root/"runs/conditioning_20261107/ECE285_conditioning_20261108_required.zip",
            "extensions.zip":root/"runs/conditioning_20261107/ECE285_conditioning_20261108_extensions_v2.zip",
            "download_ext.json":root/"artifacts/conditioning_20261107/extensions_v2/download_ext.json"}
        for name,source in paths.items():shutil.copyfile(source,fresh/name)
        download=json.loads((fresh/"download_ext.json").read_text())
        core=verify(fresh/"core.zip",CORE_SHA,"manifest.json")
        extension=verify(fresh/"extensions.zip",download["sha256"],"manifest_ext.json")
        with zipfile.ZipFile(fresh/"extensions.zip") as z:
            config=json.loads(z.read("replay_ext.json"));binding=config["locator"]["scripts/conditioning_20261107/replay_extensions.py"]
            if binding["packet"]!="extensions":raise ValueError("Bootstrap source is not the authenticated new extension")
            bootstrap=z.read(binding["member"])
            if hashlib.sha256(bootstrap).hexdigest()!=binding["sha256"]:raise ValueError("Bootstrap locator differs")
            (fresh/"replay_extensions.py").write_bytes(bootstrap)
            readme=config["locator"]["reports/conditioning_20261107/neb_pilot/README.md"]
            if readme["packet"]!="extensions" or z.read(readme["member"])!=(root/"reports/conditioning_20261107/neb_pilot/README.md").read_bytes():
                raise ValueError("Terminal NEB README differs from curated snapshot")
        command=[sys.executable,str(fresh/"replay_extensions.py"),"--core-archive",str(fresh/"core.zip"),
            "--extensions-archive",str(fresh/"extensions.zip"),"--download-ext",str(fresh/"download_ext.json"),"--output-dir",str(fresh/"results")]
        env=os.environ.copy();env.pop("PYTHONPATH",None);env["CUDA_VISIBLE_DEVICES"]=""
        result.update(replay_command=command,cwd=str(fresh),source_archives_copied_outside_CFS=True,
            member_manifests_verified_before_bootstrap=True,source_bootstrap_sha256=hashlib.sha256(bootstrap).hexdigest(),
            core_archive_sha256=CORE_SHA,extensions_archive_sha256=download["sha256"],latest_terminal_NEB_README_byte_identical=True)
        with (receipts/"fresh_replay_v2.log").open("x") as log:
            process=subprocess.run(command,cwd=fresh,env=env,stdout=log,stderr=subprocess.STDOUT)
        code=process.returncode
        if code:raise RuntimeError(f"Cached replay exited {code}; preserve detailed job logs in {fresh}/results")
        replay=json.loads((fresh/"results/replay_receipt.json").read_text())
        result.update(status="PASS",replay_receipt=str(fresh/"results/replay_receipt.json"),
            tables_compared=sum(len(j["tables"]) for j in replay["jobs"].values()),
            fields_compared=sum(t["fields_checked"] for j in replay["jobs"].values() for t in j["tables"].values()),
            all_tables_byte_identical=all(t["byte_identical"] for j in replay["jobs"].values() for t in j["tables"].values()),
            unresolved_historical_metadata={name:j["unresolved_historical_metadata"] for name,j in replay["jobs"].items() if j["unresolved_historical_metadata"]})
        shutil.copyfile(fresh/"results/replay_receipt.json",receipts/"fresh_replay_v2_receipt.json")
    except Exception as error:
        result.update(status="FAILED_PRESERVED",exception=repr(error),traceback=traceback.format_exc());code=code or 1
    backend={}
    for package in ("fairchem-core","mace-torch","matgl","chgnet","sevenn"):
        try:backend[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:backend[package]=None
    result.update(exit_code=code,backend_SDK_distributions=backend,completed_UTC=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-clock)
    (receipts/"fresh_replay_v2_execution.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2));return code


if __name__=="__main__":raise SystemExit(main())
