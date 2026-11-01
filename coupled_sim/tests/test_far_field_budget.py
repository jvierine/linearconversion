"""Independent vacuum-wave limits for the outgoing electromagnetic budget."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from coupled_arecibo import hf_port_budget

@pytest.mark.parametrize('r,t',[(0,1),(1,0),(0,0),(.3*np.exp(.7j),.8*np.exp(-.2j))])
def test_outgoing_power_includes_reflection_and_both_polarizations(r,t):
    # In vacuum the stored magnetic variables are cB. Upgoing waves have
    # cBy=Ex and cBx=-Ey; both signs reverse for downgoing waves.
    inc=np.zeros((10,8),complex);inc[0]=1;inc[1]=.7j;inc[4]=inc[0];inc[3]=-inc[1]
    reflected=inc*r;reflected[3:6]*=-1
    state=inc.copy();state[:,:4]+=reflected[:,:4];state[:,4:]=t*inc[:,4:]
    # Arbitrary local fluid amplitudes must not be mistaken for escaping EM.
    state[6:]=100+20j
    b=hf_port_budget(state,inc,1,5)
    assert b['reflected_EM_fraction']==pytest.approx(abs(r)**2,abs=1e-14)
    assert b['transmitted_EM_fraction']==pytest.approx(abs(t)**2,abs=1e-14)
    assert b['hf_nonreturn_fraction']==pytest.approx(1-abs(r)**2-abs(t)**2,abs=1e-14)
