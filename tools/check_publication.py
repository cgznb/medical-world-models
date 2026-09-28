"""Check tracked release files without printing matched sensitive values."""

from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BLOCKED_SUFFIXES = (
    ".pt", ".pth", ".ckpt", ".safetensors", ".npy", ".npz", ".nii",
    ".nii.gz", ".dcm", ".h5", ".hdf5", ".parquet", ".zip", ".tar",
    ".tar.gz", ".pem", ".key", ".pyc",
)
BLOCKED_PARTS = {
    "__pycache__", ".pytest_cache", "data", "datasets", "artifacts", "runs",
    "checkpoints", "private", "private_data",
}
RULES = {
    "private key": re.compile(r"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----"),
    "GitHub credential": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
    "deployment path": re.compile(r"/(?:home/[A-Za-z0-9_.-]+|data1/[A-Za-z0-9_.-]+|root/autodl-tmp)/"),
    "deployment host": re.compile(r"connect\.[A-Za-z0-9.-]+\.seetacloud\.com"),
    "credential in URL": re.compile(r"https?://[^\s/@:]+:[^\s/@]+@"),
}


def main() -> int:
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    files = [Path(p.decode()) for p in tracked.split(b"\0") if p]
    findings = []
    if not files:
        findings.append("No tracked files; stage the proposed release before checking.")
    for relative in files:
        path = ROOT / relative
        if path.is_symlink():
            findings.append(f"{relative}: symbolic link")
            continue
        if not path.is_file():
            findings.append(f"{relative}: missing tracked file")
            continue
        if any(part in BLOCKED_PARTS or part.startswith(".venv") for part in relative.parts):
            findings.append(f"{relative}: private/generated directory")
        if str(relative).lower().endswith(BLOCKED_SUFFIXES) or relative.name.startswith(".env"):
            findings.append(f"{relative}: blocked file type")
        if path.stat().st_size > 10 * 1024 * 1024:
            findings.append(f"{relative}: file larger than 10 MiB")
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            if path.suffix.lower() not in {".png", ".pdf"}:
                findings.append(f"{relative}: unexpected binary file")
            continue
        for name, pattern in RULES.items():
            for match in pattern.finditer(content):
                line = content.count("\n", 0, match.start()) + 1
                findings.append(f"{relative}:{line}: {name}")
    for finding in findings:
        print(finding)
    print(f"Checked {len(files)} tracked files; {len(findings)} findings.")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
