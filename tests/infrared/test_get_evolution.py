import numpy as np
import pytest

from phd_parser.infrared import IRData

WAVENUMBER_PER_CM = np.linspace(1000.0, 4000.0, 31)


@pytest.fixture
def series():
    # Each scan is a constant spectrum equal to its scan index.
    values = np.arange(6, dtype=float)[:, None] * np.ones(WAVENUMBER_PER_CM.size)
    return IRData.from_arrays(
        WAVENUMBER_PER_CM, values, tos=10.0 * np.arange(6), data_type="single_beam"
    )


def test_scalar_wavenumber_gives_a_one_dimensional_result(series):
    evolution = series.get_evolution(1600)

    assert evolution.ndim == 1
    assert evolution.dims == ("scan",)
    assert evolution.values.ndim == 1
    assert evolution.shape == (6,)
    np.testing.assert_allclose(evolution.values, np.arange(6))


def test_scalar_wavenumber_keeps_the_selected_wavenumber_as_a_scalar_coord(series):
    evolution = series.get_evolution(1600)

    assert "wavenumber" in evolution.coords
    assert evolution.coords["wavenumber"].ndim == 0
    assert float(evolution.coords["wavenumber"]) / 100.0 == pytest.approx(1600, abs=60)


def test_sequence_of_wavenumbers_still_gives_two_dimensions(series):
    evolution = series.get_evolution([1600, 2000])

    assert evolution.dims == ("scan", "wavenumber")
    assert evolution.shape == (6, 2)


def test_single_element_sequence_keeps_the_wavenumber_dimension(series):
    # A list is a sequence, so the axis is kept — only a scalar squeezes.
    evolution = series.get_evolution([1600])

    assert evolution.dims == ("scan", "wavenumber")
    assert evolution.shape == (6, 1)


def test_scalar_wavenumber_with_rolling_window(series):
    evolution = series.get_evolution(1600, rolling_window=3)

    assert evolution.dims == ("scan",)
    assert evolution.shape == (6,)


def test_scalar_result_is_directly_plottable_against_tos(series):
    evolution = series.get_evolution(1600)

    # The whole point: no [:, 0] needed to line it up with tos.
    assert evolution.values.shape == series.tos.shape


def test_tolerance_still_applies_to_a_scalar_target(series):
    with pytest.raises(ValueError, match="from the nearest grid point"):
        series.get_evolution(9999, tolerance_per_cm=1.0)


def test_raman_get_evolution_matches_the_same_convention():
    from phd_parser.raman import RamanData

    values = np.arange(4, dtype=float)[:, None] * np.ones(WAVENUMBER_PER_CM.size)
    raman = RamanData.from_arrays(
        WAVENUMBER_PER_CM, values, excitation_wavelength_nm=532.0, tos=np.arange(4, dtype=float)
    )

    assert raman.get_evolution(1600).dims == ("scan",)
    assert raman.get_evolution([1600, 2000]).dims == ("scan", "shift")
