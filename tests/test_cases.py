import pytest

from csdl_dafoam import cases


def test_list_cases_describes_each_case():
    listed = cases.list_cases()
    assert set(listed) == {"naca0012", "naca0012_euler", "naca0012_coarse"}
    assert all(listed.values())


def test_geometry_is_bundled():
    assert cases.geometry_path().is_file()
    with pytest.raises(KeyError):
        cases.geometry_path("nope")


@pytest.mark.parametrize("name", ["naca0012", "naca0012_euler", "naca0012_coarse"])
def test_copy_case_is_a_complete_openfoam_case(tmp_path, name):
    dest = cases.copy_case(name, tmp_path / "case")
    for required in (
        "0/U", "0/p", "0/T", "0/nut", "0/nuTilda",
        "system/controlDict", "system/fvSchemes", "system/fvSolution", "system/decomposeParDict",
        "constant/thermophysicalProperties", "constant/turbulenceProperties",
        "constant/polyMesh/boundary", "constant/polyMesh/points.gz", "constant/polyMesh/faces.gz",
        "constant/polyMesh/owner.gz", "constant/polyMesh/neighbour.gz",
    ):
        assert (dest / required).is_file(), required


def test_euler_case_differs_from_rans_only_in_viscosity_and_sa_seed(tmp_path):
    rans = cases.copy_case("naca0012", tmp_path / "rans")
    euler = cases.copy_case("naca0012_euler", tmp_path / "euler")

    changed = {
        str(p.relative_to(rans))
        for p in rans.rglob("*")
        if p.is_file() and p.name != cases.MARKER and p.read_bytes() != (euler / p.relative_to(rans)).read_bytes()
    }
    assert changed == {"constant/thermophysicalProperties", "0/nuTilda", "0/nut"}

    assert "mu                  1.8e-9;" in (euler / "constant/thermophysicalProperties").read_text()
    assert "mu                  0.000018;" in (rans / "constant/thermophysicalProperties").read_text()


def test_copy_case_sets_number_of_subdomains(tmp_path):
    dest = cases.copy_case("naca0012", tmp_path / "c", nprocs=2)
    text = (dest / "system" / "decomposeParDict").read_text()
    assert "numberOfSubdomains     2;" in text or "numberOfSubdomains 2;" in text.replace("  ", " ")
    assert "numberOfSubdomains     4" not in text


def test_copy_case_refuses_to_clobber(tmp_path):
    cases.copy_case("naca0012", tmp_path / "c")
    with pytest.raises(FileExistsError):
        cases.copy_case("naca0012", tmp_path / "c")
    cases.copy_case("naca0012", tmp_path / "c", overwrite=True)


def test_copy_case_validates_arguments(tmp_path):
    with pytest.raises(KeyError):
        cases.copy_case("nope", tmp_path / "c")
    with pytest.raises(ValueError):
        cases.copy_case("naca0012", tmp_path / "c", nprocs=0)


def test_prepare_case_replaces_a_previous_copy(tmp_path):
    from fakes import FakeComm

    dest = cases.prepare_case("naca0012", tmp_path / "run" / "case", comm=FakeComm())
    (dest / "processor0").mkdir()
    (dest / "100").mkdir()
    again = cases.prepare_case("naca0012", tmp_path / "run" / "case", comm=FakeComm())
    assert again == dest
    assert not (dest / "processor0").exists() and not (dest / "100").exists()


def test_prepare_case_refuses_to_delete_foreign_directories(tmp_path):
    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "thesis.tex").write_text("irreplaceable")
    with pytest.raises(FileExistsError, match="refusing"):
        cases.prepare_case("naca0012", precious)
    assert (precious / "thesis.tex").read_text() == "irreplaceable"


def test_prepare_case_without_comm_uses_one_rank(tmp_path):
    dest = cases.prepare_case("naca0012", tmp_path / "c")
    assert "numberOfSubdomains     1;" in (dest / "system" / "decomposeParDict").read_text()
