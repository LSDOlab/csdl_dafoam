"""Static guards on the Linux-VM / build scripts (they cannot be run in CI; their mistakes were costly once)."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

VM = Path(__file__).resolve().parent.parent / "scripts" / "vm"
yaml = pytest.importorskip("yaml")


def packages(text):
    text = text.replace("\\\n", " ")
    return set(re.search(r"--no-install-recommends(.*)", text).group(1).split())


def test_lima_config_keeps_its_provisioning_and_shares_nothing():
    config = yaml.safe_load((VM / "lima-dafoam.yaml").read_text())
    assert config["vmType"] == "vz" and config["arch"] == "aarch64"
    assert config["mounts"] == [], "a host mount exposes host files to the VM (this once exposed the home directory)"
    assert "{{.Dir}}" not in "".join(
        line for line in (VM / "lima-dafoam.yaml").read_text().splitlines() if not line.lstrip().startswith("#")
    )
    scripts = [p["script"] for p in config["provision"]]
    assert scripts, "provisioning section is missing: a fresh VM would have no compilers"


def test_lima_and_prereqs_script_install_the_same_packages():
    config = yaml.safe_load((VM / "lima-dafoam.yaml").read_text())
    lima = packages(config["provision"][0]["script"])
    script = packages((VM / "install_prereqs_ubuntu.sh").read_text())
    assert lima == script
    for needed in ("build-essential", "gfortran", "flex", "bison", "cmake", "libopenmpi-dev", "libcgal-dev", "python3-venv"):
        assert needed in lima


def test_build_script_is_pinned_and_ordered():
    text = (VM / "install_dafoam_ubuntu.sh").read_text()
    # sources that used to be fetched by moving branch are pinned to full commit hashes
    assert re.search(r"OpenFOAM-AD/archive/[0-9a-f]{40}\.tar\.gz", text)
    assert re.search(r"fetch -q --depth 1 origin [0-9a-f]{40}", text)
    assert "refs/heads" not in text
    assert "setuptools<70" in text  # petsc4py 3.21's build script needs it
    order = [text.index(f"stage {name} ") for name in ("1-python", "2-petsc", "2b-petsc4py", "3-openfoam", "4-openfoam-ad", "5a-pyofm", "5b-hisa4dafoam", "5-dafoam")]
    assert order == sorted(order), "Hisa4DAFoam and pyOFM must be built before DAFoam"
    traps = [line.strip() for line in text.splitlines() if line.strip().startswith("trap ")]
    assert len(traps) == 1 and traps[0].endswith(" EXIT"), "an ERR trap fires spuriously inside OpenFOAM's etc/bashrc"


def test_loader_includes_openfoam_on_resume():
    """The loader is regenerated at every start; OpenFOAM's lines must be re-added when OpenFOAM already exists."""
    text = (VM / "install_dafoam_ubuntu.sh").read_text()
    body = text[text.index("write_loader() {") : text.index("python_env() {")]
    assert "OpenFOAM-v2506/etc/bashrc" in body


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
@pytest.mark.parametrize("name", ["install_dafoam_ubuntu.sh", "install_prereqs_ubuntu.sh"])
def test_scripts_have_valid_shell_syntax(name):
    subprocess.run(["bash", "-n", str(VM / name)], check=True)


def test_lock_files_pin_the_lab_packages():
    root = VM.parent.parent
    verified = (root / "requirements-verified.txt").read_text()
    for name in ("csdl_alpha", "modopt", "idwarp-jax"):
        assert re.search(rf"^{name} @ git\+https://[^@]+@[0-9a-f]{{40}}$", verified, re.M), name
    freeze = (VM / "pip-freeze-ubuntu2404-aarch64.txt").read_text()
    assert "numpy==2.3.5" in freeze and "openmdao==" in freeze


def test_every_download_in_the_build_script_is_checksummed():
    text = (VM / "install_dafoam_ubuntu.sh").read_text()
    assert "wget" not in text.replace("wget -q \"$1\" -O \"$2\"", ""), "a download bypasses fetch()"
    fetches = re.findall(r"fetch (\S+) (\S+) ([0-9a-f]{64})", text)
    assert len(fetches) == 8
    assert len({h for _, _, h in fetches}) == 8, "two downloads share a hash: a copy-paste error"
