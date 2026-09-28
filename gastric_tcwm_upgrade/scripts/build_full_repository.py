"""Fetch the pinned original on YOUR machine, then integrate the extension.

This optional network command is not needed for standalone native-cache training.
It does not push or modify a remote repository. Git/network access is required.
"""
import argparse
from pathlib import Path
import subprocess
import zipfile
from install_into_repo import BASE_COMMIT,install


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True,help='New, nonexistent checkout directory')
    p.add_argument('--zip',type=Path,help='Optional archive of original source plus extension, without .git')
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Refusing to overwrite an existing directory')
    if args.zip and args.zip.exists():raise FileExistsError('Refusing to overwrite an existing ZIP')
    subprocess.run(['git','clone','https://github.com/cgznb/gastric-multistage-generated651.git',str(args.out)],check=True)
    subprocess.run(['git','-C',str(args.out),'checkout','--detach',BASE_COMMIT],check=True)
    install(args.out,Path(__file__).resolve().parents[1])
    if args.zip:
        args.zip.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(args.zip,'w',zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(args.out.rglob('*')):
                relative=path.relative_to(args.out)
                if path.is_file() and '.git' not in relative.parts and '__pycache__' not in relative.parts:
                    archive.write(path,Path(args.out.name)/relative)
    print(f'Created pinned original plus TCWM at {args.out}; no remote writes performed.')

if __name__=='__main__':main()
