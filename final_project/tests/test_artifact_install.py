import json,subprocess,sys,zipfile
from pathlib import Path
import pytest
from cathode_cvae.artifact_fetch import extract_verified


def test_verified_extraction_preserves_local_changes_and_rejects_escape(tmp_path):
 p=tmp_path/'run.zip'
 with zipfile.ZipFile(p,'w') as z:z.writestr('data/input.txt','original')
 out=tmp_path/'out';assert extract_verified(p,out)==1;assert extract_verified(p,out)==1
 (out/'data/input.txt').write_text('collaborator change')
 with pytest.raises(FileExistsError):extract_verified(p,out)
 assert (out/'data/input.txt').read_text()=='collaborator change'
 with zipfile.ZipFile(p,'w') as z:z.writestr('../escape','bad')
 with pytest.raises(ValueError):extract_verified(p,out)
 assert not (tmp_path/'escape').exists()


def test_selected_installer_profiles_without_installation(tmp_path):
 script=Path(__file__).resolve().parents[1]/'scripts/install.sh'
 command=['bash',str(script),'--python',sys.executable,'--prefix',str(tmp_path/'not-created'),'--dry-run']
 result=subprocess.run(command+['--backends','uma,mace,uma'],text=True,capture_output=True,check=True)
 assert json.loads(result.stdout)['profiles']==['learning','uma','mace']
 assert not (tmp_path/'not-created').exists()
 result=subprocess.run(command+['--core-only'],text=True,capture_output=True,check=True)
 assert json.loads(result.stdout)['profiles']==['learning']
 assert subprocess.run(command+['--backends','unknown'],capture_output=True).returncode!=0
