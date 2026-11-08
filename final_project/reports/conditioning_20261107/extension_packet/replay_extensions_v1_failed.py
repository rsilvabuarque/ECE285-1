"""Verify both release ZIPs and replay optional scalar tables on a fresh host.

No calculator, foundation weight, optimizer, or CFS default is needed. Required
core sources are restored byte-for-byte, then the separately versioned editor
modules are added. Original protocols/receipts remain byte-identical. Exact
verified path strings in parsed configurations are redirected to a new mirror;
the explicit relocation receipt distinguishes that runtime view from originals.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import stat
import subprocess
import sys
import zipfile

CORE_SHA256 = "b720c932edf7f48cf1d8789f39abe9135ca9f9b17a5322b4558545d3243a72df"
RELEASE_TAG = "ece285-conditioning-20261108"
R = "runs/conditioning_20261107"
S = "scripts/conditioning_20261107"
FORBIDDEN = ("fairchem", "mace", "matgl", "chgnet", "sevenn", "tensorflow")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024**2), b""): h.update(block)
    return h.hexdigest()


def safe_relative(value):
    p = PurePosixPath(value)
    if not p.parts or p.is_absolute() or ".." in p.parts or any(":" in x or "\\" in x for x in p.parts):
        raise ValueError(f"Unsafe archive-relative path: {value}")
    return p.as_posix()


def verify_archive(path, expected_sha256, manifest_name):
    """Authenticate outer bytes before trusting a complete, safe member list."""
    if len(expected_sha256) != 64 or digest(path) != expected_sha256: raise ValueError("Release archive SHA256 differs")
    with zipfile.ZipFile(path) as z:
        names = [info.filename for info in z.infolist()]
        if len(set(names)) != len(names): raise ValueError("Duplicate ZIP member")
        for info in z.infolist():
            if safe_relative(info.filename) != info.filename or info.is_dir(): raise ValueError("ZIP must contain only safe file members")
            if stat.S_ISLNK(info.external_attr >> 16): raise ValueError("ZIP symlinks are not permitted")
        manifest = json.loads(z.read(manifest_name)); files = manifest.get("files")
        if not isinstance(files, dict) or not files or manifest_name in files: raise ValueError("Malformed member manifest")
        if set(names) != set(files) | {manifest_name}: raise ValueError("ZIP members differ from manifest")
        for name, expected in files.items():
            safe_relative(name); info = z.getinfo(name)
            if info.file_size != expected["bytes"]: raise ValueError(f"Member byte count differs: {name}")
            h = hashlib.sha256()
            with z.open(name) as f:
                for block in iter(lambda:f.read(8*1024**2), b""): h.update(block)
            if h.hexdigest() != expected["sha256"]: raise ValueError(f"Member SHA256 differs: {name}")
    return manifest


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def compare_csv(actual, expected):
    """Exact identities/counts/order; floating arithmetic allows only roundoff.

    Eight binary64 epsilon-scaled units in the column's own units are an
    engineering replay allowance, never a physical accuracy or success gate.
    """
    from itertools import zip_longest
    fields = 0
    with Path(actual).open(newline="") as aa, Path(expected).open(newline="") as bb:
        for row, pair in enumerate(zip_longest(csv.reader(aa), csv.reader(bb))):
            left, right = pair
            if left is None or right is None or len(left) != len(right): raise ValueError(f"CSV shape changed: {Path(actual).name} row {row}")
            for col, (x,y) in enumerate(zip(left,right)):
                fields += 1
                if x == y: continue
                if row == 0 or not (any(c in x for c in ".eE") and any(c in y for c in ".eE")):
                    raise ValueError(f"CSV identity/count changed: {Path(actual).name} row {row} col {col}")
                try: a,b = float(x),float(y)
                except ValueError as e: raise ValueError("CSV text differs") from e
                tolerance = 8*sys.float_info.epsilon*max(1.,abs(a),abs(b))
                if not math.isfinite(a) or not math.isfinite(b) or abs(a-b)>tolerance:
                    raise ValueError(f"CSV floating value changed: {Path(actual).name} row {row} col {col}")
    return dict(fields_checked=fields, byte_identical=digest(actual)==digest(expected), actual_sha256=digest(actual), reference_sha256=digest(expected))


class Relocator:
    """A path-only parsed-configuration view; hashes/labels stay untouched."""
    def __init__(self, original_root, mirror, locator):
        self.original_root = Path(original_root); self.mirror = Path(mirror).resolve()
        self.mapping = {str(self.original_root/relative):str(self.mirror/safe_relative(relative)) for relative in locator}
        self.bindings={str(self.original_root/relative):binding for relative,binding in locator.items()}
        self.by_hash=defaultdict(list)
        for relative,binding in locator.items():
            if "sha256" in binding:self.by_hash[binding["sha256"]].append(str(self.mirror/safe_relative(relative)))
        self.versioned_overrides={}
        self.used = set()

    def replace(self, value):
        if isinstance(value, str) and value in self.mapping:
            self.used.add(value); return self.mapping[value]
        if isinstance(value, list): return [self.replace(x) for x in value]
        if isinstance(value, dict):
            # A mutable historical filename can denote several snapshots. Its
            # recorded SHA, rather than the most recent or first path alias,
            # determines which authenticated original bytes the runtime sees.
            path=value.get("path");expected=value.get("sha256",value.get("sha256_at_freeze"))
            if isinstance(path,str) and path in self.mapping and isinstance(expected,str) and len(expected)==64:
                binding=self.bindings[path]
                if binding.get("sha256")==expected:target=self.mapping[path]
                elif self.by_hash.get(expected):target=sorted(self.by_hash[expected])[0]
                else:
                    raise ValueError(f"Historical source version is absent from both packets: {path} SHA256 {expected}")
                self.versioned_overrides[(path,expected)]=target
                return {k:(target if k=="path" else self.replace(v)) for k,v in value.items()}
            return {k:self.replace(v) for k,v in value.items()}
        return value

    def path(self, value):
        text = str(value)
        if text in self.mapping: self.used.add(text); return Path(self.mapping[text])
        path = Path(value)
        # Never allow an unverified original CFS path to fall back to a live
        # project file just because replay happens to run on that same host.
        if path.is_absolute() and path.is_relative_to(self.original_root) and not path.is_relative_to(self.mirror.parent):
            raise ValueError(f"Unbound historical runtime path: {path}")
        return path

    def json_proxy(self):
        owner = self
        class Proxy:
            def loads(self, text, *args, **kwargs): return owner.replace(json.loads(text,*args,**kwargs))
            def __getattr__(self, name): return getattr(json,name)
        return Proxy()

    def numpy_proxy(self, numpy):
        owner=self
        class Proxy:
            def load(self, path, *args, **kwargs):
                return numpy.load(owner.path(path) if isinstance(path,(str,os.PathLike)) else path,*args,**kwargs)
            def __getattr__(self,name):return getattr(numpy,name)
        return Proxy()


def load_script(root, name, relocation=None):
    path = Path(root)/S/(name+".py")
    spec = importlib.util.spec_from_file_location("replay_verified_"+name,path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    if relocation is not None:
        # These are explicit path adapters; sha256 still hashes actual original
        # bytes copied from authenticated releases. No expected hash is faked.
        module.json = relocation.json_proxy()
        if hasattr(module,"sha256"): module.sha256 = lambda p:digest(relocation.path(p))
        if hasattr(module,"np"):module.np=relocation.numpy_proxy(module.np)
        if hasattr(module,"Path"):module.Path=relocation.path
    return module


def tables(actual, reference, names):
    return {name:compare_csv(Path(actual)/name,Path(reference)/name) for name in names}


def with_argv(args, function):
    old = sys.argv; sys.argv = [old[0],*map(str,args)]
    try: return function()
    finally: sys.argv = old


def cached_cost_adapter(root, inventory, summarize):
    """Recompute query counts; raw shard hashes are verified inventory evidence.

    Portable scalar replay cannot rehash arrays deliberately retained on HPC.
    It instead verifies the authenticated curation inventory against original
    task JSON and recomputes the same identity accounting. Output JSON states
    this narrower scope; primary query_accounting.csv is checked unchanged.
    """
    import numpy as np
    lookup = {x["task_id"]:x for x in inventory["raw_shards"]}
    def audit(generated, oracle, before_path, rows, output):
        before=json.loads(Path(before_path).read_text())
        registry=Path(root)/R/"collected/optional_final/registry_snapshot.json"
        current=json.loads(registry.read_text())
        cache=Path(oracle)/"cache_uma.npz"; certificate=json.loads(cache.with_suffix(".json").read_text())
        if digest(cache)!=certificate["npz_sha256"]: raise ValueError("Archived scalar cache differs from original certificate")
        with np.load(cache,allow_pickle=False) as z:cached=set(z["structure_ids"].astype(str))
        pre={v["structure_id"] for v in before["physical_keys"].values() if v["model"]=="uma"}; available=pre|cached
        pool=Path(generated)/"candidates.npz"; pool_sha=digest(pool)
        with np.load(pool,allow_pickle=False) as z:ids=set(z["structure_ids"].astype(str))
        fresh={v["structure_id"] for k,v in current["physical_keys"].items() if k not in before["physical_keys"]
            and v["model"]=="uma" and v["input_sha256"]==pool_sha and v["provenance"]=="new_scoring"}
        if fresh!=ids-available:raise ValueError("Portable editor stage attribution differs")
        completed=set()
        for task in current["tasks"]:
            if task["model"]!="uma" or task["input_sha256"]!=pool_sha:continue
            record=json.loads((Path(oracle)/"scores/uma"/(task["task_id"]+".json")).read_text())
            binding=lookup[task["task_id"]]
            if record["status"]!="complete" or record["task"]!=task or record["output_sha256"]!=binding["sha256"]:
                raise ValueError("Task receipt differs from authenticated raw-shard inventory")
            if completed&set(task["expected_structure_ids"]):raise ValueError("Repeated attributed task identities")
            completed.update(task["expected_structure_ids"])
        if completed!=fresh:raise ValueError("Attributed completion coverage differs")
        result=[]
        for scope,rr in [("all_requested_bins",rows),("low_requests_only",[r for r in rows if r["requested_bin"]=="low"])]:
            seeds={r["seed_scoring_id"] for r in rr};children={r["structure_id"] for r in rr if summarize.truth(r["energy_known"])}
            sets=[("shared_seed_verification",seeds)]+[(m,{r["structure_id"] for r in rr if r["method"]==m and summarize.truth(r["energy_known"])}) for m in ("learned_edit","mutation")]+[("union_children",children),("union_seeds_and_children",seeds|children)]
            for name,ss in sets:
                result.append(dict(scope=scope,role_or_method=name,unique_geometry_queries=len(ss),pre_editor_registry_hits=len(ss&pre),
                    pre_editor_certified_cache_hits=len(ss&cached),union_preexisting_exact_labels=len(ss&available),
                    standalone_fresh_queries_after_required=len(ss-available),actual_editor_stage_fresh_ids_encountered=len(ss&fresh),
                    caution="Method rows share labels; do not sum them. Seed verification is a shared overhead, not charged twice."))
        summarize.write_csv(Path(output)/"query_accounting.csv",result)
        record=dict(status="CACHED_IDENTITY_ACCOUNTING_REPLAY",raw_array_bytes_rehashed_offsite=False,
            task_receipts_and_curation_inventory_verified=True,new_MLIP_calls=0,
            distinction="Replay recomputes exact geometry identity counts; original HPC raw-shard hash/stat audit is archived separately.")
        write_json(Path(output)/"cost_audit.json",record);return available,record
    return audit


def run_job(job, config, mirror, output, relocation):
    """All scientific calculation uses verified, unchanged saved source files."""
    root=Path(mirror);output=Path(output);jobs=config["jobs"]
    from cathode_cvae.runtime import require_compute_allocation
    require_compute_allocation()
    if job=="editor":
        module=load_script(root,"evaluate_edit_comparison",relocation)
        with_argv(["--root",root,"--generated",root/jobs[job]["generated"],"--scores",root/config["scores"]["uma"],"--output-dir",output],module.main)
    elif job=="aggregate":
        from cathode_cvae.condition_evaluation import evaluate
        generated=root/jobs[job]["generated"]
        receipt=evaluate(generated/"attempts.csv",generated/"candidates.npz",
            {t:root/config["scores"][t] for t in ("uma","mace")},
            {t:root/f"artifacts/overnight_20261107/data/condition_schema_{t}.json" for t in ("uma","mace")},
            root/R/"data/training_keys.json",generated/"history.json",root/"artifacts/current/reports/generation_thresholds.csv",output,
            root/"artifacts/current/core/data/parents/O3_N48_pilot.npz")
        if receipt["status"]!="PASS":raise ValueError("Aggregate scalar replay incomplete")
    elif job=="temperature":
        # analyze() insists that its new analysis subdirectory not exist. Keep
        # the original folder intact and create an explicitly derived view.
        source=root/jobs[job]["generated"];view=output.parent/"temperature_inputs";view.mkdir()
        for name in ("protocol.json","history.json","registered_history.json","reference_diagnostics.csv","requested_allocations.csv","finite12_support.json"):
            shutil.copyfile(source/name,view/name)
        shutil.copytree(source/"generated",view/"generated")
        module=load_script(root,"temperature_followup",relocation)
        module.analyze(view,root/config["scores"]["uma"])
        shutil.move(str(view/"analysis"),output)
    elif job=="editor_low":
        module=load_script(root,"summarize_editor",relocation)
        inventory=json.loads((root.parent/"raw_shards_inventory.json").read_text())
        module.cost_audit=cached_cost_adapter(root,inventory,module)
        # The unchanged source requires evaluated inputs beneath --root for its
        # provenance receipt. This authenticated original table was already
        # compared against the freshly replayed editor table in the first job.
        with_argv(["--root",root,"--evaluated",root/jobs["editor"]["reference"],"--generated",root/jobs["editor"]["generated"],"--output-dir",output],module.main)
    elif job=="editor_posthoc":
        output.mkdir()
        for name in ("attempt_outcomes.csv","paired_energy_outcomes.csv","receipt.json"):
            shutil.copyfile(output.parent/"editor"/name,output/name)
        source=root/"reports/conditioning_20261107/independent_editor_metrics/summarize_request_energy.py"
        spec=importlib.util.spec_from_file_location("verified_editor_posthoc",source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.main(["--directory",str(output)])
    elif job=="aggregate_independent":
        module=load_script(root,"check_aggregate_prior",relocation)
        module.main(["--root",str(root),"--output-dir",str(output)])
    elif job=="neb":
        if jobs[job]["status"]=="DEFERRED":
            output.mkdir();write_json(output/"receipt.json",dict(status="DEFERRED_AS_RECORDED",source=jobs[job]["deferred_receipt"],new_model_calls=0))
            return dict(status="DEFERRED_AS_RECORDED",tables={})
        module=load_script(root,"neb_summarize",relocation)
        module.main(["--run-root",str(root/jobs[job]["run"]),"--output-dir",str(output)])
        return dict(status="PASS",tables=tables(output,root/jobs[job]["reference"],["outcomes.csv"]))
    else:raise ValueError("Unknown replay job")
    checked=tables(output,root/jobs[job]["reference"],jobs[job]["tables"])
    if any(name.split(".")[0] in FORBIDDEN for name in sys.modules):raise ValueError("Backend SDK imported during scalar replay")
    return dict(status="PASS",tables=checked)


def materialize(core_archive, extensions_archive, config, destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(core_archive) as core,zipfile.ZipFile(extensions_archive) as ext:
        for relative,binding in sorted(config["locator"].items()):
            path=destination/safe_relative(relative);path.parent.mkdir(parents=True,exist_ok=True)
            archive={"core":core,"extensions":ext}.get(binding["packet"])
            if archive is None:raise ValueError("Unknown source archive")
            with archive.open(safe_relative(binding["member"])) as source,path.open("xb") as target:shutil.copyfileobj(source,target)
            if digest(path)!=binding["sha256"] or path.stat().st_size!=binding["bytes"]:raise ValueError("Materialized source binding differs")
        (destination.parent/"raw_shards_inventory.json").write_bytes(ext.read("raw_shards_inventory.json"))
    return destination


def training_command(config, mirror, fit_name, output, device):
    tasks=json.loads((Path(mirror)/config["training_tasks"]).read_text())["tasks"]
    selected=[t for t in tasks if t["id"]==fit_name]
    if len(selected)!=1:raise ValueError("Choose one of the three preserved editor fit IDs")
    relocate=Relocator(config["original_root"],mirror,config["locator"]);argv=selected[0]["argv"][:]
    for i,token in enumerate(argv):
        if i and argv[i-1]=="--root":argv[i]=str(Path(mirror).resolve())
        elif i and argv[i-1]=="--output-dir":argv[i]=str(Path(output).resolve())
        elif i and argv[i-1]=="--device":argv[i]=device
        elif token in relocate.mapping:argv[i]=str(relocate.path(token))
    return [sys.executable,"-m",selected[0]["trainer_module"],*argv]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--core-archive",type=Path);p.add_argument("--extensions-archive",type=Path);p.add_argument("--download-ext",type=Path)
    p.add_argument("--output-dir",type=Path);p.add_argument("--verify-only",action="store_true")
    p.add_argument("--print-training-command");p.add_argument("--training-output",type=Path);p.add_argument("--device",default="cuda")
    p.add_argument("--_run-job",type=Path,help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if a._run_job:
        job=json.loads(a._run_job.read_text());config=json.loads(Path(job["config"]).read_text())
        relocation=Relocator(config["original_root"],job["mirror"],config["locator"])
        result=run_job(job["name"],config,job["mirror"],job["output"],relocation)
        result.update(new_MLIP_calls=0,new_training_updates=0,parsed_path_overrides={k:relocation.mapping[k] for k in sorted(relocation.used)},
            versioned_path_overrides=[dict(original_path=p,sha256=h,relocated_path=target) for (p,h),target in sorted(relocation.versioned_overrides.items())],
            backend_SDKs_imported=[x for x in sys.modules if x.split(".")[0] in FORBIDDEN])
        write_json(job["receipt"],result);return 0
    if not all((a.core_archive,a.extensions_archive,a.download_ext)):p.error("Supply --core-archive --extensions-archive --download-ext")
    download=json.loads(a.download_ext.read_text())
    if download.get("core_archive_sha256")!=CORE_SHA256:raise ValueError("Extension is bound to another core ZIP")
    core=verify_archive(a.core_archive,CORE_SHA256,"manifest.json")
    ext=verify_archive(a.extensions_archive,download["sha256"],"manifest_ext.json")
    if core.get("release_tag")!=RELEASE_TAG or core.get("core_full_fit_count")!=28 or ext.get("core_archive_sha256")!=CORE_SHA256:
        raise ValueError("Release identity differs")
    with zipfile.ZipFile(a.core_archive) as z:
        if hashlib.sha256(z.read("manifest.json")).hexdigest()!=ext.get("core_manifest_sha256"):
            raise ValueError("Extension core-manifest binding differs")
    with zipfile.ZipFile(a.extensions_archive) as z:config=json.loads(z.read("replay_ext.json"))
    if a.verify_only:
        print(json.dumps(dict(status="VERIFIED",core_files=len(core["files"]),extension_files=len(ext["files"]),new_MLIP_calls=0,new_training_updates=0)));return 0
    if a.output_dir is None:p.error("--output-dir is required")
    a.output_dir=a.output_dir.resolve();a.core_archive=a.core_archive.resolve();a.extensions_archive=a.extensions_archive.resolve()
    if a.output_dir.exists():raise FileExistsError("Use a fresh replay directory; originals are never overwritten")
    a.output_dir.mkdir(parents=True)
    mirror=materialize(a.core_archive,a.extensions_archive,config,a.output_dir/"verified_project")
    for relative,expected in {**core["source_hashes"],**ext["source_hashes"]}.items():
        if digest(mirror/relative)!=expected:raise ValueError(f"Verified runtime source differs: {relative}")
    config_path=a.output_dir/"verified_replay_config.json";write_json(config_path,config)
    if a.print_training_command:
        if a.training_output is None:p.error("--training-output required when printing a command")
        command=training_command(config,mirror,a.print_training_command,a.training_output,a.device)
        print("PYTHONPATH="+shlex.quote(str(mirror/"src"))+" "+shlex.join(command));return 0
    env=os.environ.copy();env.update(PYTHONPATH=str(mirror/"src"),CUDA_VISIBLE_DEVICES="",MPLBACKEND="Agg")
    results={};script=mirror/S/"replay_extensions.py"
    for name in ("editor","aggregate","aggregate_independent","temperature","editor_low","editor_posthoc","neb"):
        job_path=a.output_dir/(name+"_job.json");receipt=a.output_dir/(name+"_replay.json")
        write_json(job_path,dict(name=name,config=str(config_path.resolve()),mirror=str(mirror.resolve()),output=str((a.output_dir/name).resolve()),receipt=str(receipt.resolve())))
        with (a.output_dir/(name+".log")).open("w") as log:
            subprocess.run([sys.executable,str(script),"--_run-job",str(job_path.resolve())],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=mirror)
        results[name]=json.loads(receipt.read_text())
    result=dict(status="PASS",completed_UTC=datetime.now(timezone.utc).isoformat(),core_archive_sha256=CORE_SHA256,
        extensions_archive_sha256=download["sha256"],scope="Cached table and paired-energy reproduction, not new sampling, fitting or physical validation",
        original_protocols_and_receipts_unchanged=True,path_relocation_scope="Exact verified path strings in parsed runtime configurations only; original bytes/hashes are preserved",
        jobs=results,new_MLIP_calls=0,new_training_updates=0,raw_EFS_scope="Archived curation hash/stat inventory and task receipts; raw arrays remain at HPC",
        tolerance="Exact integer/categorical/order; eight binary64 epsilon-scaled units for changed float text only",
        verified_runtime_source_hashes={relative:digest(mirror/relative) for relative in sorted({**core["source_hashes"],**ext["source_hashes"]})})
    write_json(a.output_dir/"replay_receipt.json",result)
    print(json.dumps(dict(status="PASS",jobs=list(results),new_MLIP_calls=0,new_training_updates=0),indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
