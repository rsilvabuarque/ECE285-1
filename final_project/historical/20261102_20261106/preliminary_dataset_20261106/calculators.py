"""Five conservative ASE calculators with explicit task and stress conventions."""
import importlib.metadata as md
import hashlib,json,os,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def load(name):
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    paths=json.loads((HERE/'checkpoints/paths.json').read_text())
    t=time.perf_counter()
    if name=='uma':
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit
        from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
        settings=InferenceSettings(tf32=False,nitro=False,merge_mole=False,compile=False,
            execution_mode='general',base_precision_dtype='float32',internal_graph_gen_version=2,
            activation_checkpointing=False)
        calc=FAIRChemCalculator(load_predict_unit(paths[name],inference_settings=settings,device='cuda'),task_name='omat')
        detail=dict(package='fairchem-core',task='omat',settings=str(settings))
    elif name=='mace':
        from mace.calculators import MACECalculator
        calc=MACECalculator(model_paths=paths[name],device='cuda',default_dtype='float32',head='omat_pbe',enable_cueq=False,compile_mode=None)
        detail=dict(package='mace-torch',task='omat_pbe')
    elif name=='sevennet':
        from sevenn.calculator import SevenNetCalculator
        calc=SevenNetCalculator(model=paths[name],modal='mpa',device='cuda',enable_cueq=False,enable_flash=False,enable_oeq=False,enable_lunatp=False)
        detail=dict(package='sevenn',task='mpa')
    else:
        import matgl
        matgl.set_backend('PyG');matgl.set_default_dtype('float',32)
        from matgl.ext.ase import PESCalculator
        potential=matgl.load_model(paths[name]).to(device='cuda',dtype=torch.float32).eval()
        assert potential.calc_forces and potential.calc_stresses
        calc=PESCalculator(potential=potential,stress_unit='eV/A3',stress_weight=1.,use_voigt=True)
        detail=dict(package='matgl',task='MatPES-PBE',stress_unit='eV/A3')
    torch.cuda.synchronize()
    p=Path(paths[name]);files=[p] if p.is_file() else sorted(q for q in p.iterdir() if q.suffix in ['.json','.pt'])
    hashes={}
    for q in files:
        h=hashlib.sha256()
        with q.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
        hashes[q.name]=h.hexdigest()
    return calc,dict(model=name,checkpoint=paths[name],checkpoint_hashes=hashes,dtype='float32',tf32=False,
        package_version=md.version(detail['package']),torch=torch.__version__,ase=md.version('ase'),
        gpu=torch.cuda.get_device_name(),load_seconds=time.perf_counter()-t,
        job_id=os.environ.get('SLURM_JOB_ID'),host=os.environ.get('SLURMD_NODENAME'),
        gpu_visibility=os.environ.get('CUDA_VISIBLE_DEVICES'),**detail)

def evaluate(calc,atoms):
    atoms.calc=calc
    energy=float(atoms.get_potential_energy());forces=np.array(atoms.get_forces());stress=np.array(atoms.get_stress())
    assert forces.shape==(len(atoms),3) and stress.shape==(6,)
    assert all(np.isfinite(a).all() for a in [energy,forces,stress])
    return energy,forces,stress
