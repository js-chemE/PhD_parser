"""CasaXPS-style peak fitting: lineshapes, constraints, the Mo 3d fixture and the stored result."""
from pathlib import Path

import numpy as np
import pytest

from phd_parser.spectroscopy.xps import XPSData
from phd_parser.spectroscopy.xps import fitting
from phd_parser.spectroscopy.xps.fitting import (
    DOUBLETS,
    Lineshape,
    PeakSpec,
    doublet,
    fit_peaks,
    gaussian_lorentzian_product,
    gaussian_lorentzian_sum,
    lorentzian_asymmetric,
    parse_lineshape,
)

FIXTURES = Path(__file__).parent
AVG_MO3D = FIXTURES / "avantage" / "Mo3d Scan.avg"
AVG_C1S = FIXTURES / "avantage" / "C1s Scan.avg"
CASA_DOUBLET = FIXTURES / "casaxps_export_doublet.txt"

FINE = np.linspace(-80.0, 80.0, 320_001)


@pytest.mark.parametrize("text", ["GL(30)", "SGL(30)", "LA(1.53,243)", "LA(1.2,1.5,100)", "GL(0)", "GL(100)"])
def test_every_lineshape_integrates_to_its_area(text):
    curve = parse_lineshape(text).evaluate(FINE, 0.0, 1.0, 2.5)
    # the Lorentzian tails beyond ±80 widths hold a few tenths of a percent at most
    assert np.trapezoid(curve, FINE) == pytest.approx(2.5, rel=5e-3)


def test_gaussian_lorentzian_forms_match_casaxps_definitions():
    reduced = FINE
    gaussian = np.exp(-4 * np.log(2) * reduced**2)
    lorentzian = 1 / (1 + 4 * reduced**2)
    product = gaussian_lorentzian_product(FINE, 0.0, 1.0, 1.0, 0.3)
    total = gaussian_lorentzian_sum(FINE, 0.0, 1.0, 1.0, 0.3)
    expected_product = gaussian**0.7 / (1 + 4 * 0.3 * reduced**2)
    expected_sum = 0.7 * gaussian + 0.3 * lorentzian
    np.testing.assert_allclose(product / product.max(), expected_product, atol=1e-12)
    np.testing.assert_allclose(total / total.max(), expected_sum, atol=1e-12)
    # the sum has the FWHM F, the product is a little narrower (CasaXPS's definition)
    half = np.abs(FINE[total >= total.max() / 2])
    assert 2 * half.max() == pytest.approx(1.0, abs=1e-3)
    half = np.abs(FINE[product >= product.max() / 2])
    assert 2 * half.max() < 1.0


def test_lorentzian_asymmetric_is_normalised_and_tails_to_high_binding_energy():
    symmetric = lorentzian_asymmetric(FINE, 0.0, 1.0, 1.0, 1.53, 1.53, 243)
    assert np.trapezoid(symmetric, FINE) == pytest.approx(1.0, rel=2e-3)
    np.testing.assert_allclose(symmetric, symmetric[::-1], atol=1e-6)
    half = np.abs(FINE[symmetric >= symmetric.max() / 2])
    assert 2 * half.max() == pytest.approx(1.0, abs=2e-3)  # fwhm is the width of the final curve
    # alpha < beta: the smaller exponent, on the high binding energy side, gives the longer tail
    asymmetric = lorentzian_asymmetric(FINE, 0.0, 1.0, 1.0, 1.0, 3.0, 50)
    assert np.trapezoid(asymmetric, FINE) == pytest.approx(1.0, rel=1e-2)
    at = lambda x: float(np.interp(x, FINE, asymmetric))  # noqa: E731
    assert at(2.0) > 10 * at(-2.0)
    assert at(0.75) > at(-0.75)
    # no convolution for n = 0, and an exponent at or below 1/2 has no finite area
    bare = lorentzian_asymmetric(np.array([0.0, 0.5]), 0.0, 1.0, 1.0, 1.0, 1.0, 0)
    assert bare[1] == pytest.approx(bare[0] / 2)
    with pytest.raises(ValueError):
        lorentzian_asymmetric(FINE, 0.0, 1.0, 1.0, 0.5, 1.0, 0)


def test_parse_lineshape_round_trips_the_casaxps_notation():
    for text in ("GL(30)", "SGL(30)", "LA(1.53,243)", "LA(1.2,1.5,100)"):
        shape = parse_lineshape(text)
        assert shape.casaxps == text and str(shape) == text
        assert parse_lineshape(shape) is shape
    assert parse_lineshape("gl(30)") == Lineshape("GL", lorentzian_fraction=0.3)
    la = parse_lineshape("LA(1.53,243)")
    assert la.kind == "LA" and la.alpha == la.beta == 1.53 and la.gaussian_width_index == 243
    for bad in ("Voigt(30)", "GL(30,40)", "LA(0.5)", "GL(130)", "LA(0.4,1,10)"):
        with pytest.raises(ValueError):
            parse_lineshape(bad)


def test_single_gl_peak_is_recovered_within_one_percent():
    energy = np.arange(280.0, 292.0, 0.1)
    truth = gaussian_lorentzian_product(energy, 285.2, 1.4, 3200.0, 0.3)
    noise = np.random.default_rng(1).normal(0.0, 5.0, energy.size)
    result = fit_peaks(energy, truth + noise, [PeakSpec("C 1s", 285.0, fwhm_eV=1.0)])
    assert result.success and result.n_free_parameters == 3
    peak = result.peaks[0]
    assert peak.position_eV == pytest.approx(285.2, rel=1e-2)
    assert peak.fwhm_eV == pytest.approx(1.4, rel=1e-2)
    assert peak.area == pytest.approx(3200.0, rel=1e-2)
    assert peak.area_fraction == 1.0 and peak.lineshape == "GL(30)"
    assert peak.height == pytest.approx(truth.max(), rel=2e-2)
    assert 0 < peak.area_error < 0.05 * peak.area and 0 < peak.position_error_eV < 0.05
    assert result.reduced_chi_square == pytest.approx(25.0, rel=0.3)
    frame = result.to_frame()
    assert list(frame["name"]) == ["C 1s"] and "area_fraction" in frame.columns
    assert "C 1s" in result.report() and "reduced chi-square" in result.report()
    np.testing.assert_allclose(result.envelope, result.peak_curves.sum(axis=0))


def test_doublet_with_fixed_splitting_and_ratio_is_recovered():
    energy = np.arange(226.0, 240.0, 0.1)
    shape = parse_lineshape("LA(1.53,243)")
    truth = shape.evaluate(energy, 232.6, 1.1, 9000.0) + shape.evaluate(energy, 235.73, 1.1, 6000.0)
    noise = np.random.default_rng(2).normal(0.0, 8.0, energy.size)
    specs = doublet("Mo(VI) 3d", 232.3, lineshape="LA(1.53,243)", **DOUBLETS["Mo 3d"])
    assert [spec.name for spec in specs] == ["Mo(VI) 3d5/2", "Mo(VI) 3d3/2"]
    assert specs[1].position_offset_from == ("Mo(VI) 3d5/2", 3.13)
    assert specs[1].area_ratio_to == ("Mo(VI) 3d5/2", pytest.approx(2 / 3))
    assert specs[1].fwhm_equal_to == "Mo(VI) 3d5/2"
    result = fit_peaks(energy, truth + noise, specs)
    assert result.success and result.n_free_parameters == 3  # the second peak is fully derived
    first, second = result.peaks
    assert first.position_eV == pytest.approx(232.6, abs=0.02)
    assert second.position_eV == pytest.approx(first.position_eV + 3.13)
    assert first.fwhm_eV == pytest.approx(1.1, rel=1e-2) and second.fwhm_eV == first.fwhm_eV
    assert second.area == pytest.approx(first.area * 2 / 3)
    assert first.area == pytest.approx(9000.0, rel=1e-2)
    assert first.area_fraction == pytest.approx(0.6) and second.area_fraction == pytest.approx(0.4)
    assert second.position_error_eV == first.position_error_eV
    assert second.area_error == pytest.approx(first.area_error * 2 / 3)
    # the caller's specs are not touched (the area guess is filled in on a copy)
    assert specs[0].area is None


def test_constraints_are_checked():
    energy = np.arange(280.0, 290.0, 0.1)
    values = gaussian_lorentzian_product(energy, 285.0, 1.0, 100.0, 0.3)
    with pytest.raises(ValueError, match="unique"):
        fit_peaks(energy, values, [PeakSpec("a", 285.0), PeakSpec("a", 286.0)])
    with pytest.raises(ValueError, match="refers to"):
        fit_peaks(energy, values, [PeakSpec("a", 285.0, fwhm_equal_to="b")])
    with pytest.raises(ValueError, match="Circular"):
        fit_peaks(energy, values, [PeakSpec("a", 285.0, fwhm_equal_to="b"), PeakSpec("b", 286.0, fwhm_equal_to="a")])
    with pytest.raises(ValueError):
        PeakSpec("a", 285.0, fwhm_bounds_eV=(2.0, 1.0))
    with pytest.raises(ValueError):
        fit_peaks(energy, values, [])


def test_mo3d_fixture_fit_is_mostly_molybdenum_six():
    # calibrate on the adventitious carbon of the same sample, then fit two Mo 3d doublets
    carbon = XPSData.from_avantage_avg(AVG_C1S).fit_peaks([PeakSpec("C 1s", 284.3, fwhm_eV=1.5)], min_eV=280, max_eV=292)
    region = XPSData.from_avantage_avg(AVG_MO3D).calibrate_binding_energy(284.8, float(carbon.peak_table()["position_eV"].iloc[0]))
    six = doublet("Mo(VI) 3d", 232.6, lineshape="LA(1.53,243)", **DOUBLETS["Mo 3d"])
    five = doublet("Mo(V) 3d", 231.3, lineshape="LA(1.53,243)", **DOUBLETS["Mo 3d"])
    # CasaXPS-style constraints for a minor species: fixed chemical shift and the width of the major one
    five[0].position_offset_from = ("Mo(VI) 3d5/2", -1.3)
    five[0].fwhm_equal_to = "Mo(VI) 3d5/2"
    fitted = region.fit_peaks(six + five, background="shirley", min_eV=226, max_eV=240)

    assert fitted.has_fit and fitted.has_background and fitted.envelope is not None
    assert fitted.ds["peaks"].shape == (4, fitted.n_points)
    assert fitted.ds.attrs["analysis_source"] == "phd_parser"
    summary = fitted.ds.attrs["fit"]
    assert summary["success"] and summary["n_free_parameters"] == 4 and summary["background"]["method"] == "shirley"
    assert [spec["name"] for spec in summary["specs"]] == ["Mo(VI) 3d5/2", "Mo(VI) 3d3/2", "Mo(V) 3d5/2", "Mo(V) 3d3/2"]
    assert fitted.history[-1]["step"] == "fit_peaks"
    assert region.has_fit is False  # immutable

    table = fitted.peak_table()
    assert list(table["name"]) == ["Mo(VI) 3d5/2", "Mo(VI) 3d3/2", "Mo(V) 3d5/2", "Mo(V) 3d3/2"]
    by_name = table.set_index("name")
    assert by_name.loc["Mo(VI) 3d5/2", "position_eV"] == pytest.approx(232.3, abs=0.3)
    assert by_name.loc["Mo(VI) 3d3/2", "position_eV"] == pytest.approx(by_name.loc["Mo(VI) 3d5/2", "position_eV"] + 3.13)
    assert by_name.loc["Mo(V) 3d5/2", "position_eV"] == pytest.approx(by_name.loc["Mo(VI) 3d5/2", "position_eV"] - 1.3)
    assert 1.5 < by_name.loc["Mo(VI) 3d5/2", "fwhm_eV"] < 2.3  # the fixture is a broad, charging spectrum
    assert by_name["fwhm_eV"].nunique() == 1
    assert by_name.loc["Mo(VI) 3d3/2", "area"] == pytest.approx(by_name.loc["Mo(VI) 3d5/2", "area"] * 2 / 3)
    assert by_name["area_fraction"].sum() == pytest.approx(1.0)
    assert (table["lineshape"] == "LA(1.53,243)").all()
    assert table["position_error_eV"].notna().all()
    mo_six = by_name.loc[["Mo(VI) 3d5/2", "Mo(VI) 3d3/2"], "area_fraction"].sum()
    assert mo_six > 0.8  # 0.84 on this fixture; see the module report for why not more

    # the curves are stored like a CasaXPS export: on top of the background, envelope ≈ data in the window
    window = (fitted.binding_energy >= 226) & (fitted.binding_energy <= 240)
    net = fitted.envelope[window] - fitted.background[window]
    residual = fitted.values[window] - fitted.envelope[window]
    assert np.sqrt(np.mean(residual**2)) < 0.04 * net.max()
    np.testing.assert_allclose(fitted.ds["peaks"].values.sum(axis=0), fitted.envelope + 3 * fitted.background)
    np.testing.assert_allclose(fitted.values, region.values)  # data untouched, the fit sits on the background
    peak_area = np.trapezoid(fitted.ds["peaks"].values[0] - fitted.background, fitted.binding_energy)
    assert peak_area == pytest.approx(by_name.loc["Mo(VI) 3d5/2", "area"], rel=0.05)


def test_fit_without_background_and_poisson_weights():
    region = XPSData.from_avantage_avg(AVG_MO3D).subtract_background("shirley", 226, 240)
    specs = doublet("Mo 3d", 231.9, lineshape="GL(30)", **DOUBLETS["Mo 3d"])
    fitted = region.fit_peaks(specs, background=None, min_eV=226, max_eV=240, weights="poisson")
    assert fitted.ds.attrs["fit"]["background"]["included_in_curves"] is False
    assert fitted.ds.attrs["fit"]["background"]["weights"] == "poisson"
    np.testing.assert_allclose(fitted.envelope, fitted.ds["peaks"].values.sum(axis=0))
    assert fitted.peak_table()["area_fraction"].iloc[0] == pytest.approx(0.6)
    with pytest.raises(ValueError):
        region.fit_peaks(specs, background=None, weights=np.ones(3))


def test_peak_table_works_for_casaxps_imports():
    table = XPSData.from_casaxps_export(CASA_DOUBLET).peak_table()
    assert list(table["name"]) == ["Mo 3d5/2", "Mo 3d3/2"]
    assert table["area_fraction"].sum() == pytest.approx(1.0)
    assert table["area_fraction"].iloc[0] == pytest.approx(7710.55 / (7710.55 + 5140.37))
    assert (table["height"] > 0).all() and "position_error_eV" not in table.columns
    with pytest.raises(ValueError):
        XPSData.from_avantage_avg(AVG_MO3D).peak_table()


def test_netcdf_round_trip_keeps_the_fit(tmp_path):
    specs = doublet("Mo 3d", 231.9, lineshape="LA(1.53,243)", **DOUBLETS["Mo 3d"])
    fitted = XPSData.from_avantage_avg(AVG_MO3D).fit_peaks(specs, min_eV=226, max_eV=240)
    fitted.to_netcdf(tmp_path / "mo3d_fit.nc")
    back = XPSData.from_netcdf(tmp_path / "mo3d_fit.nc")
    assert back.has_fit and back.has_background
    for column in ("name", "position_eV", "fwhm_eV", "area", "area_fraction", "lineshape"):
        np.testing.assert_array_equal(back.peak_table()[column].values, fitted.peak_table()[column].values)
    np.testing.assert_allclose(back.ds["peaks"].values, fitted.ds["peaks"].values)
    np.testing.assert_allclose(back.envelope, fitted.envelope)
    assert back.ds.attrs["fit"] == fitted.ds.attrs["fit"]
    assert [entry["step"] for entry in back.history][-2:] == ["fit_peaks", "from_netcdf"]
    # a refit replaces the stored peaks instead of piling up coordinates
    again = back.fit_peaks(specs, min_eV=226, max_eV=240)
    assert again.ds["peaks"].shape == (2, again.n_points) and again.history[-1]["step"] == "fit_peaks"


# ----------------------------------------------------------------
# ranged relations, width ratio, LA(a), information criteria, end-point averaging
# ----------------------------------------------------------------

def test_la_with_one_number_is_la_a_a_0():
    shape = fitting.parse_lineshape("LA(50)")
    assert shape.alpha == shape.beta == 50 and shape.gaussian_width_index == 0 and shape.casaxps == "LA(50)"
    energy = np.linspace(-10, 10, 4001)
    assert np.trapezoid(shape.evaluate(energy, 0.0, 1.5, 2.0), energy) == pytest.approx(2.0, rel=1e-3)


def test_ranged_offset_and_width_ratio_are_recovered():
    energy = np.linspace(450, 472, 1101)
    truth = (fitting.gaussian_lorentzian_product(energy, 458.6, 1.2, 100.0, 0.3)
             + fitting.gaussian_lorentzian_product(energy, 458.6 + 5.72, 1.2 * 1.45, 50.0, 0.3))
    specs = [
        fitting.PeakSpec("2p3/2", 458.4, 1.0),
        fitting.PeakSpec("2p1/2", 464.0, 1.0, position_offset_from=("2p3/2", (5.5, 5.9)),
                         area_ratio_to=("2p3/2", 0.5), fwhm_ratio_to=("2p3/2", (1.0, 2.0))),
    ]
    result = fitting.fit_peaks(energy, truth, specs)
    second = {p.name: p for p in result.peaks}["2p1/2"]
    first = {p.name: p for p in result.peaks}["2p3/2"]
    assert second.position_eV - first.position_eV == pytest.approx(5.72, abs=1e-3)
    assert second.fwhm_eV / first.fwhm_eV == pytest.approx(1.45, rel=1e-3)
    assert result.n_free_parameters == 5


def test_bic_prefers_the_true_number_of_peaks():
    rng = np.random.default_rng(1)
    energy = np.linspace(225, 240, 751)
    truth = fitting.gaussian_lorentzian_product(energy, 232.6, 1.4, 100.0, 0.3) + rng.normal(0, 0.05, energy.size)
    one = fitting.fit_peaks(energy, truth, [fitting.PeakSpec("a", 232.5, 1.2)])
    two = fitting.fit_peaks(energy, truth, [fitting.PeakSpec("a", 232.5, 1.2), fitting.PeakSpec("b", 230.0, 1.2, area=1.0)])
    assert one.bic < two.bic


def test_relation_range_must_be_ordered():
    with pytest.raises(ValueError):
        fitting.PeakSpec("b", 1.0, position_offset_from=("a", (2.0, 1.0)))
    with pytest.raises(ValueError):
        fitting.PeakSpec("b", 1.0, fwhm_equal_to="a", fwhm_ratio_to=("a", 1.5))


def test_end_point_averaging_damps_a_noisy_edge():
    energy = np.linspace(0, 10, 501)
    values = 10 + 5 * np.exp(-((energy - 5) ** 2)) + np.where(energy > 9.99, 3.0, 0.0)   # one hot point at the high end
    data = XPSData.from_arrays(energy, values, source_energy_eV=1486.68)
    single = data.subtract_background("linear")
    averaged = data.subtract_background("linear", average_eV=0.5)
    assert single.background[-1] == pytest.approx(13.0)
    assert averaged.background[-1] < 10.2
