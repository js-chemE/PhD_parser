import pytest
from phd_parser.xrd.xrd_e1290 import read_xy_e1290
from phd_parser.xrd.core import XRDData
import numpy as np
import numpy.typing as npt
import os

@pytest.fixture
def mock_intensity_data():
    return np.array([100, 200, 150, 300, 250])

@pytest.fixture
def mock_angle_data():
    return np.array([10.0, 20.0, 30.0, 40.0, 50.0])

@pytest.fixture
def sample_e1290_path():
    return os.path.join(os.path.dirname(__file__), 'xrd_mock_e1290.xy')

def test_read_xrd_e1290(sample_e1290_path):
    # Read the data without normalization
    xrd_data = read_xy_e1290(sample_e1290_path, normalize=False)
    assert isinstance(xrd_data, np.ndarray)
    assert xrd_data.shape[0] == 2  # Expecting two rows: 2-theta and intensity

    # Read the data with normalization
    xrd_data_normalized = read_xy_e1290(sample_e1290_path, normalize=True)
    assert np.isclose(xrd_data_normalized[1].max(), 1.0)  # Max intensity should be 1 after normalization

def test_xrd(mock_angle_data, mock_intensity_data):
    angle = mock_angle_data
    intensity = mock_intensity_data
    # Create a mock XRD instance
    xrd_instance = XRD(angle=angle, intensity=intensity)

    assert isinstance(xrd_instance, XRD)
    assert np.array_equal(xrd_instance.angle, angle)
    assert np.array_equal(xrd_instance.intensity, intensity)

def test_xrd_mismatched_lengths(mock_angle_data, mock_intensity_data): # Intentionally mismatched lengths
    angle = mock_angle_data
    intensity = mock_intensity_data[:-1]
    # Test for mismatched angle and intensity lengths
    with pytest.raises(ValueError):
        XRD(angle=angle, intensity=intensity)

def test_xrd_from_e1290():
    # Path to a sample XRD file for testing
    sample_file = os.path.join(os.path.dirname(__file__), 'xrd_mock_e1290.xy')

    # Create XRD instance from the sample file
    xrd_instance = XRD.from_e1290(sample_file)

    assert isinstance(xrd_instance, XRD)
    assert xrd_instance.angle.size > 0
    assert xrd_instance.intensity.size > 0