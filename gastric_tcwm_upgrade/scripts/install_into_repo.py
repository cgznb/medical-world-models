"""Add TCWM to an existing Generated651 checkout without replacing baseline code."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess

BASE_COMMIT='aec8cbe08157687fd447594c8a9406d6ee5a464d'


def install(target: Path, source: Path, *, allow_different_commit=False):
    target,source=target.resolve(),source.resolve()
    required=['src/stageworld/generated700_models.py','src/stageworld/data/treatment_compact.py',
              'src/stageworld/surgery_s2_models.py']
    if not all((target/name).is_file() for name in required):
        raise ValueError('Target is not a compatible Generated651 source checkout')
    if source==target or target.is_relative_to(source):
        raise ValueError('Keep the extension source separate from the target checkout')
    head=None
    if (target/'.git').exists():
        head=subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
        if head!=BASE_COMMIT and not allow_different_commit:
            raise ValueError(f'Target commit {head} differs from audited {BASE_COMMIT}; inspect changes before explicit override')
    dest=target/'extensions/tcwm'
    launcher=target/'scripts/run_tcwm.py'
    entryreadme=target/'README_TCWM.md'
    if dest.exists() or launcher.exists() or entryreadme.exists():
        raise FileExistsError('Refusing to overwrite an existing TCWM installation; use a clean checkout')
    old_hashes={name:hashlib.sha256((target/name).read_bytes()).hexdigest() for name in required}
    shutil.copytree(source,dest,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','.git','*.egg-info','*.pyc','artifacts','outputs'))
    launcher.parent.mkdir(parents=True,exist_ok=True)
    launcher.write_text('''from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
sys.path.insert(0,str(root/'extensions/tcwm/src'))
from stageworld_tcwm.cli import main
if __name__=='__main__':
    main()
''')
    entryreadme.write_text('# TCWM research extension\n\nSee `extensions/tcwm/README.md` and `extensions/tcwm/docs/ALGORITHM_ZH.md`.\n\nNew entry point: `python scripts/run_tcwm.py --help`. Original baseline files are unchanged.\n')
    assert old_hashes=={name:hashlib.sha256((target/name).read_bytes()).hexdigest() for name in required}
    report={'audited_base_commit':BASE_COMMIT,'installed_target_commit':head,'baseline_files_unchanged':old_hashes,
            'extension_path':str(dest),'remote_repository_modified':False}
    (dest/'installation.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True,type=Path)
    p.add_argument('--allow-different-commit',action='store_true')
    a=p.parse_args()
    print(json.dumps(install(a.repo,Path(__file__).resolve().parents[1],allow_different_commit=a.allow_different_commit),indent=2))

if __name__=='__main__':main()
