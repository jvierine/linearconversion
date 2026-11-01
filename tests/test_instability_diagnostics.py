import pytest
from isr.instability_diagnostics import otsi_homogeneous_field, otsi_optimistic_growth


def test_reproduces_djuth_gonzales_homogeneous_threshold():
    # Their 110 km example: ne=3.2e11, Te=Ti=230 K, nu=18000 s^-1.
    assert otsi_homogeneous_field(3.2e11,230,230,18000) == pytest.approx(.72,rel=.015)


def test_reproduces_djuth_gonzales_optimistic_iron_growth():
    gamma=otsi_optimistic_growth(2.2,18,5.1e6,56)
    assert gamma == pytest.approx(12200,rel=.02)
    assert 1/(2*gamma) == pytest.approx(41e-6,rel=.02)


def test_power_and_field_conventions():
    base=otsi_homogeneous_field(3.2e11,230,230,18000)
    assert otsi_homogeneous_field(3.2e11,230,230,18000,60) == pytest.approx(2*base)
    assert otsi_optimistic_growth(8,18,5.1e6,30) == pytest.approx(4*otsi_optimistic_growth(1,18,5.1e6,30))
