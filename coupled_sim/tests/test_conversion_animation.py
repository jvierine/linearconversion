"""Check the actual portable native recordings, not a plotted wave cartoon."""
from pathlib import Path
import h5py,numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]/'reference/coupled/conversion_animation'

@pytest.mark.parametrize('name,sign',[('topside_optimum',1),('bottomside_optimum',-1),('topside_spitze',1)])
def test_native_wave_direction_frequency_and_attenuation(name,sign):
    with h5py.File(ROOT/(name+'.h5')) as h:
        t=h['time_s'][:];z=h['z_relative_m'][:]
        q=h['density_real_ppm'][:]+1j*h['density_imag_ppm'][:]
        E=h['Ez_real_ratio'][:]+1j*h['Ez_imag_ratio'][:]
        om=2*np.pi*h.attrs['frequency_hz']
        # A wrong time sign, frozen movie, or synthesized propagation direction
        # fails these direct phase tests on native recordings.
        rot=np.exp(1j*om*(t-t[0]))[:,None]
        for field in [q,E]:
            relative=np.linalg.norm(field*rot-field[0])/np.linalg.norm(field)
            assert relative<1e-7
        kz=np.gradient(np.unwrap(np.angle(q[0])),z)
        window=(sign*z>1)&(sign*z<8)
        assert np.mean(sign*kz[window]>0)>.99
        assert abs(np.interp(sign*8,z,kz))>abs(np.interp(sign*2,z,kz))
        assert sign*h.attrs['bragg_offset_m']>0
        amp=abs(q[0])
        assert np.interp(sign*12,z,amp)<.2*np.interp(sign*2,z,amp)


def test_spitze_is_transmission_not_conversion_optimum():
    with h5py.File(ROOT/'scan.h5') as h:
        t=h['angle_deg'][:];j=np.argmin(abs(t+h.attrs['spitze_magnitude_deg']))
        assert h['transmitted_EM_fraction'][j]>.9
        assert max(h['upward_pressure_fraction'][:])>20*h['upward_pressure_fraction'][j]
    with h5py.File(ROOT/'validation.h5') as h:
        for key in ['topside_optimum','bottomside_optimum','topside_spitze']:
            a=h[key].attrs
            assert a['amplitude_mesh_relative_error']<.02
            assert a['Gauss_relative_error']<1e-6
            assert a['bragg_relative_error']<.05
            assert a['outward_pressure_flux_at_4m']>0
