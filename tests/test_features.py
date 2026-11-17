"""Unit tests for the elementary feature maths."""
import numpy as np
import pytest

from cs2cd.features import aim_error, angular_speed, view_vector, wrap_deg


def test_wrap_deg_handles_the_180_boundary():
    assert wrap_deg(179 - (-179)) == pytest.approx(-2.0)   # 358 deg -> -2 deg
    assert wrap_deg(-350) == pytest.approx(10.0)
    assert wrap_deg(0) == 0


def test_angular_speed_crossing_yaw_boundary_is_small():
    yaw = np.array([179.0, -179.0])        # a 2 degree turn, not 358
    pitch = np.array([0.0, 0.0])
    assert angular_speed(pitch, yaw)[0] == pytest.approx(2.0)


def test_angular_speed_combines_pitch_and_yaw():
    w = angular_speed(np.array([0.0, 3.0]), np.array([0.0, 4.0]))
    assert w[0] == pytest.approx(5.0)


def test_view_vector_is_unit_and_points_down_for_positive_pitch():
    v = view_vector(np.array([30.0]), np.array([90.0]))[0]
    assert np.linalg.norm(v) == pytest.approx(1.0)
    assert v[2] < 0 and v[1] > 0


def test_aim_error_zero_when_looking_at_target():
    eye = np.array([[0.0, 0.0, 0.0]])
    target = np.array([[100.0, 0.0, 0.0]])
    assert aim_error(np.array([0.0]), np.array([0.0]), eye, target)[0] == pytest.approx(0.0, abs=1e-4)


def test_aim_error_ninety_degrees_when_looking_sideways():
    eye = np.array([[0.0, 0.0, 0.0]])
    target = np.array([[100.0, 0.0, 0.0]])
    assert aim_error(np.array([0.0]), np.array([90.0]), eye, target)[0] == pytest.approx(90.0)
