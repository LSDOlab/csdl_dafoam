import pytest

from csdl_dafoam import doctor


def test_check_reports_every_component_and_the_foam_environment(monkeypatch):
    monkeypatch.delenv("WM_PROJECT_DIR", raising=False)
    monkeypatch.delenv("DAFOAM_ROOT_PATH", raising=False)
    rows = {label: ok for label, ok, _, _ in doctor.check()}
    for label in ("NumPy", "SciPy", "DAFoam", "IDWarp-JAX (LSDOlab's, from git)", "IDWarp (Fortran)", "qpsolvers", "OpenFOAM environment", "simpleFoam on PATH"):
        assert label in rows
    assert rows["NumPy"] is True
    assert rows["OpenFOAM environment"] is False


def test_foam_environment_detected_from_variables(monkeypatch):
    monkeypatch.setenv("WM_PROJECT_DIR", "/opt/openfoam")
    monkeypatch.setenv("DAFOAM_ROOT_PATH", "/opt/dafoam")
    rows = {label: (ok, detail) for label, ok, detail, _ in doctor.check()}
    assert rows["OpenFOAM environment"] == (True, "/opt/openfoam")
    assert rows["DAFOAM_ROOT_PATH"] == (True, "/opt/dafoam")


def test_main_exit_status(monkeypatch, capsys):
    monkeypatch.setattr(doctor, "check", lambda: [("DAFoam", False, "not installed", "DAFoam runs")])
    assert doctor.main([]) == 1
    assert doctor.main(["--no-fail"]) == 0
    assert "DAFoam" in capsys.readouterr().out


def test_broken_install_is_reported_not_raised(monkeypatch):
    # find_spec says it exists but importing it explodes (e.g. a missing shared library)
    monkeypatch.setattr(doctor, "_COMPONENTS", [("os", "fake", "everything")])
    monkeypatch.setattr(doctor, "_version", lambda name: (None, "ImportError: libfoo.so not found"))
    row = doctor.check()[0]
    assert row[:3] == ("fake", False, "ImportError: libfoo.so not found")


def test_fortran_idwarp_is_optional(monkeypatch):
    rows = [("IDWarp (Fortran)", False, "not installed", "optional: warper='idwarp'")]
    monkeypatch.setattr(doctor, "check", lambda: rows)
    assert doctor.main([]) == 0  # only the default stack is required


def test_missing_default_warper_is_required(monkeypatch):
    rows = [("IDWarp-JAX (LSDOlab's, from git)", False, "not installed", "mesh warping (default)")]
    monkeypatch.setattr(doctor, "check", lambda: rows)
    assert doctor.main([]) == 1
