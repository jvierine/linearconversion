"""A grazing, evanescent source is not an incident propagating radio channel."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from incident_port import incoming_channels

def test_unmagnetized_propagation_cutoff():
    # n^2=1-wp^2/w^2=0.64: transverse k/w must be below 0.8.
    for q in [0.1,0.5,0.79]:
        channels=incoming_channels(1.,np.zeros(3),.6,.001,q)
        assert len(channels)==2
        assert all(mode[2]>0 for mode in channels)
    for q in [.81,.95]:
        assert incoming_channels(1.,np.zeros(3),.6,.001,q)==[]
