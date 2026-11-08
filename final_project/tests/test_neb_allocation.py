"""Mock-only checks: no Slurm step, model, or allocation is launched."""
import importlib.util
import json
from pathlib import Path
import signal
import subprocess

import pytest

path=Path(__file__).resolve().parents[1]/"scripts/conditioning_20261107/neb_allocation.py"
spec=importlib.util.spec_from_file_location("neb_allocation",path)
controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)


def test_two_distinct_event_commands_and_one_declared_refinement(tmp_path):
    config=dict(project_root=str(tmp_path),events=["first","second"],uma_python="/python/uma",
        prepared="/prepared",uma_checkpoint="/weights",run_root="/results")
    first,second=[controller.worker_command(config,i) for i in range(2)]
    assert "--refine" in first and "--refine" not in second
    assert first[first.index("--event")+1]=="first"
    assert second[second.index("--event")+1]=="second"
    assert first[first.index("--budget-seconds")+1]=="2400"
    step=controller.srun_prefix("123",2)
    assert step[step.index("-n")+1]=="2"
    assert "--gpus-per-task=1" in step and "--gpu-bind=single:1" in step
    assert "salloc" not in step and "sbatch" not in step


def test_profile_requires_successful_marked_uma_path_and_wall_reserve(tmp_path):
    folder=tmp_path/"first";folder.mkdir()
    (folder/"receipt.json").write_text(json.dumps(dict(climbing_band=dict(converged=True),
        endpoint_event_validation=dict(valid=True),forward_barrier_eV=.5)))
    assert controller.profile_eligible(tmp_path,"first",750)[0]
    assert not controller.profile_eligible(tmp_path,"first",749)[0]
    (folder/"receipt.json").write_text(json.dumps(dict(climbing_band=dict(converged=True),
        endpoint_event_validation=dict(valid=False),forward_barrier_eV=None)))
    assert not controller.profile_eligible(tmp_path,"first",1000)[0]


def test_timeout_terminates_only_owned_subprocess_group_and_preserves_receipt(tmp_path,monkeypatch):
    class FakeProcess:
        pid=123456
        def __init__(self):self.calls=0
        def wait(self,timeout):
            self.calls+=1
            if self.calls<3:raise subprocess.TimeoutExpired(["mock"],timeout)
            return -9
    process=FakeProcess();created=[];signals=[]
    def create(argv,**kwargs):created.append((argv,kwargs));return process
    monkeypatch.setattr(controller.subprocess,"Popen",create)
    monkeypatch.setattr(controller.os,"killpg",lambda group,sig:signals.append((group,sig)))
    result=controller.run_bounded(["mock","--no-real-step"],1,tmp_path/"mock.log",{},tmp_path)
    assert result["timed_out"] and result["forced_kill_after_grace"] and result["returncode"]==-9
    assert signals==[(123456,signal.SIGTERM),(123456,signal.SIGKILL)]
    assert created[0][1]["start_new_session"] is True
    assert (tmp_path/"mock.log").is_file()


def test_worker_signal_exception_cleans_its_child_and_writes_interrupted_receipt(tmp_path,monkeypatch):
    class InterruptedProcess:
        pid=987654
        def __init__(self):self.waits=0
        def wait(self,timeout):
            self.waits+=1
            if self.waits==1:raise InterruptedError("synthetic worker SIGTERM")
            return -15
    process=InterruptedProcess();signals=[]
    monkeypatch.setattr(controller.subprocess,"Popen",lambda *a,**k:process)
    monkeypatch.setattr(controller.os,"killpg",lambda group,sig:signals.append((group,sig)))
    monkeypatch.setenv("SLURM_PROCID","0");monkeypatch.setenv("SLURM_JOB_ID","mock-job")
    monkeypatch.setattr(controller,"digest",lambda _:"mock-source-sha")
    config=dict(project_root=str(tmp_path),events=["first","second"],uma_python="/mock/python",
        prepared="/prepared",uma_checkpoint="/weights",run_root=str(tmp_path))
    assert controller.run_worker(config)==2
    record=json.loads((tmp_path/"controller_worker_0.json").read_text())
    assert record["status"]=="WORKER_INTERRUPTED_OR_FAILED"
    assert record["exception_type"]=="InterruptedError"
    assert signals==[(987654,signal.SIGTERM)]


def test_raising_handlers_are_installed_for_worker_branch(monkeypatch):
    handlers={}
    monkeypatch.setattr(controller.signal,"signal",lambda name,handler:handlers.setdefault(name,handler))
    controller.install_signal_handlers()
    with pytest.raises(InterruptedError):handlers[signal.SIGTERM](signal.SIGTERM,None)
    with pytest.raises(InterruptedError):handlers[signal.SIGINT](signal.SIGINT,None)


def test_validation_preserves_distinct_venv_python_symlinks_and_checkpoint_aliases(tmp_path):
    project=tmp_path/"project";scripts=project/"scripts/conditioning_20261107";scripts.mkdir(parents=True)
    (scripts/"neb_run.py").write_text("# synthetic path fixture, never executed\n")
    base=tmp_path/"base_python";base.write_text("unused fixture")
    weight=tmp_path/"weight_blob";weight.write_text("unused fixture")
    config=dict(project_root=str(project),prepared=str(tmp_path/"prepared"),run_root=str(tmp_path/"results"),
        summary_output=str(tmp_path/"summary"))
    for model in ("uma","mace","summary"):
        python=tmp_path/model/"bin/python";python.parent.mkdir(parents=True);python.symlink_to(base)
        config[model+"_python"]=str(python)
    for model in ("uma","mace"):
        checkpoint=tmp_path/model/(model+"_checkpoint.pt");checkpoint.symlink_to(weight)
        config[model+"_checkpoint"]=str(checkpoint)
    prepared=Path(config["prepared"]);prepared.mkdir()
    (prepared/"protocol.json").write_text(json.dumps(dict(maximum_events=2,per_event_wall_seconds=2400,
        optional_MACE_profile_wall_seconds=600,two_GPU_window_maximum_wall_seconds=3600,
        events=[dict(event_id="first"),dict(event_id="second")],refinement=dict(event_id="first"))))
    validated=controller.validate_config(config)
    for name in ("uma_python","mace_python","summary_python","uma_checkpoint","mace_checkpoint"):
        assert validated[name]==config[name]
        assert Path(validated[name]).is_symlink()
    assert validated["uma_python"]!=validated["mace_python"]
    assert controller.worker_command(validated,0)[0]==config["uma_python"]
