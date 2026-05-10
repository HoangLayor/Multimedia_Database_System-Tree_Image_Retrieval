import pytest
import numpy as np
import cv2
from src.features.geometry import extract_geometry
from src.features.color import extract_color
from src.features.texture import extract_texture
from src.features.shape import extract_shape
from src.features.extractor import extract

@pytest.fixture
def dummy_tree_image():
    """Creates a 512x512 dummy tree image (green circle on black background)."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    # Green canopy
    cv2.circle(img, (256, 200), 100, (0, 200, 0), -1)
    # Brown trunk
    cv2.rectangle(img, (240, 300), (272, 450), (40, 70, 140), -1)
    return img

def test_extract_geometry(dummy_tree_image):
    features = extract_geometry(dummy_tree_image)
    assert len(features) == 5
    assert all(isinstance(f, float) for f in features)
    # H/W ratio should be roughly (250+50)/200 = 1.5
    assert features[0] > 1.0
    # Density should be between 0 and 1
    assert 0 < features[1] < 1.0
    # Symmetry should be high for a centered circle/rect
    assert features[3] > 0.8

def test_extract_color(dummy_tree_image):
    features = extract_color(dummy_tree_image)
    assert len(features) == 9
    # Primary Green Hue should be in the green range (~60/180 = 0.33)
    assert 0.2 < features[0] < 0.5
    # Green saturation should be high
    assert features[1] > 0.5
    # Histogram bins should sum to something positive
    assert sum(features[4:]) > 0

def test_extract_texture(dummy_tree_image):
    features = extract_texture(dummy_tree_image)
    assert len(features) == 4
    # All between 0 and 1 (roughly normalized)
    assert all(0 <= f <= 1.0 for f in features)

def test_extract_shape(dummy_tree_image):
    features = extract_shape(dummy_tree_image)
    assert len(features) == 9
    # Solidity for a circle + rect should be high (now at index 7)
    assert features[7] > 0.7
    # Circularity for this shape should be less than 1.0 (now at index 8)
    assert features[8] < 1.0

def test_full_extractor(tmp_path, dummy_tree_image):
    img_path = tmp_path / "test_tree.jpg"
    cv2.imwrite(str(img_path), dummy_tree_image)
    
    features = extract(img_path)
    assert features.shape == (27,)
    assert features.dtype == np.float32
    assert not np.any(np.isnan(features))
