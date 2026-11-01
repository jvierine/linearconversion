import numpy as np
import pytest
from scipy.constants import e,m_e,epsilon_0
from isr.driven import forced_density,matching_density
from isr.kinetic import radar_k
from isr.suprathermal import energy_distribution,plasma_line,lorentzian

def test_cold_frequency_density_matching():
 n=matching_density(5e6,radar_k(),0)
 assert n==pytest.approx((2*np.pi*5e6)**2*epsilon_0*m_e/e**2)
 assert matching_density(5e6,radar_k(),1000)<n

def test_forced_response_collective_shielding():
 k=radar_k();chi=4.;ext=.001
 # Poisson self consistency for fixed ions, no independent density source.
 dn=forced_density(k,chi,1+chi,ext)
 phi_self=-e*dn/(epsilon_0*k*k)
 assert ext+phi_self==pytest.approx(ext/(1+chi))

def test_growing_mode_has_no_stationary_power():
 E=np.arange(1,100.001,.05);p=energy_distribution(E)
 ne=(2*np.pi*6e6)**2*epsilon_0*m_e/e**2
 r=plasma_line(ne,1500,35e-6,30,E,p)
 assert r['damping_s']<0
 assert not r['stable'] and np.isnan(r['area'])
 with pytest.raises(ValueError,match='stationary'):
  lorentzian([r['f_hz']],r)
