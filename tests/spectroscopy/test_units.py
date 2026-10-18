"""The spectral axis in any unit: one stored wavenumber, every other reading derived from it."""
import numpy as np
import pytest
from scipy import constants as const

from phd_parser.spectroscopy import IRData, RamanData, SpectroscopyData, units

AXIS_PER_CM = np.array([400.0, 1000.0, 2000.0, 4000.0])


def test_every_unit_round_trips():
    wavenumber_si = AXIS_PER_CM * 100.0
    for unit in units.SPECTRAL_UNITS:
        back = units.to_wavenumber(units.from_wavenumber(wavenumber_si, unit), unit)
        np.testing.assert_allclose(back, wavenumber_si, rtol=1e-12, err_msg=unit)


def test_known_conversions():
    assert units.from_wavenumber(100.0 * 100.0, "um") == pytest.approx(100.0)        # 100 cm^-1 is 100 um
    assert units.from_wavenumber(4000.0 * 100.0, "nm") == pytest.approx(2500.0)
    assert units.from_wavenumber(8065.544 * 100.0, "eV") == pytest.approx(1.0, rel=1e-6)  # 1 eV is 8065.54 cm^-1
    assert units.from_wavenumber(1.0 * 100.0, "kJ mol^-1") == pytest.approx(0.011963, rel=1e-4)
    assert units.from_wavenumber(1.0 * 100.0, "THz") == pytest.approx(const.c * 100.0 / 1e12)
    assert units.from_wavenumber(8.065544e6 * 100.0, "keV") == pytest.approx(1.0, rel=1e-5)
    assert units.normalise_unit("cm-1") == "cm^-1" and units.normalise_unit("kJ/mol") == "kJ mol^-1"
    assert units.normalise_unit("µm") == "um" and units.quantity_of("Å") == "wavelength"
    with pytest.raises(ValueError, match="Unknown spectral unit"):
        units.from_wavenumber(1.0, "furlong")


def test_axis_of_a_spectrum_in_any_unit():
    ir = IRData.from_arrays(AXIS_PER_CM, np.ones(4))
    np.testing.assert_allclose(ir.axis(), AXIS_PER_CM)
    np.testing.assert_allclose(ir.axis("m^-1"), ir.wavenumber)
    np.testing.assert_allclose(ir.axis("um"), 1e4 / AXIS_PER_CM)
    np.testing.assert_allclose(ir.axis("eV"), ir.energy_eV)
    np.testing.assert_allclose(ir.axis("kJ/mol"), ir.energy_kJ_per_mol)
    assert ir.axis_label("um") == "wavelength / um"
    assert ir.axis_label("kJ mol^-1") == "molar photon energy / kJ mol^-1"


def test_from_arrays_takes_the_axis_in_any_unit():
    by_wavelength = SpectroscopyData.from_arrays(1e4 / AXIS_PER_CM, np.ones(4), axis_unit="um", technique="infrared")
    np.testing.assert_allclose(by_wavelength.wavenumber_per_cm, AXIS_PER_CM)
    by_energy = SpectroscopyData.from_arrays(AXIS_PER_CM * 100.0 * const.Planck * const.c / const.electron_volt,
                                             np.ones(4), axis_unit="eV")
    np.testing.assert_allclose(by_energy.wavenumber_per_cm, AXIS_PER_CM)


def test_raman_shift_reads_as_shift_or_as_scattered_light():
    raman = RamanData.from_arrays(AXIS_PER_CM, np.ones(4), excitation_wavelength_nm=532.0)
    np.testing.assert_allclose(raman.axis("cm^-1"), AXIS_PER_CM)                  # the shift
    np.testing.assert_allclose(raman.axis("meV"), AXIS_PER_CM * 100.0 * const.Planck * const.c / const.electron_volt * 1e3)
    np.testing.assert_allclose(raman.axis("nm"), 1e7 / (1e7 / 532.0 - AXIS_PER_CM))  # the scattered light
    np.testing.assert_allclose(raman.axis("nm"), raman.scattered_wavelength_nm)
    assert raman.axis_label("cm^-1") == "raman shift / cm^-1"
    assert raman.axis_label("nm") == "scattered wavelength / nm"
    # a spectrometer's wavelength axis stored as the shift
    from_light = SpectroscopyData.from_arrays(raman.axis("nm"), np.ones(4), axis_unit="nm", excitation_wavelength_nm=532.0)
    np.testing.assert_allclose(from_light.wavenumber_per_cm, AXIS_PER_CM)
    bare = SpectroscopyData.from_arrays(AXIS_PER_CM, np.ones(4), technique="raman")
    with pytest.raises(ValueError, match="excitation_wavelength_nm"):
        bare.axis("nm")


def test_select_axis_range_in_any_unit():
    axis = np.linspace(400.0, 4000.0, 361)
    ir = IRData.from_arrays(axis, np.ones(axis.size))
    mid_infrared = ir.select_axis_range(2.5, 25.0, "um")
    assert mid_infrared.wavenumber_per_cm.min() == pytest.approx(400.0) and mid_infrared.wavenumber_per_cm.max() == pytest.approx(4000.0)
    assert ir.select_axis_range(0.1, 0.2, "eV").n_spectral == ir.select_wavenumber_range(806.5544, 1613.1088).n_spectral
    assert ir.select_axis_range(maximum=5.0, unit="um").wavenumber_per_cm.min() == pytest.approx(2000.0)
    assert ir.select_axis_range(minimum=5.0, unit="um").wavenumber_per_cm.max() == pytest.approx(2000.0)
    assert ir.select_axis_range(1000, 2000).history[-1]["step"] == "select_axis_range"
