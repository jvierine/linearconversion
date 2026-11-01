"""Fetch exact independent ISR implementations named by the validation submodule."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'isr_spec_validation'))
from compare import source,JUHA_REV,JOHN_REV
source('https://github.com/jvierine/isr_analysis.git',JUHA_REV,'vierinen')
source('https://github.com/jswoboda/ISRSpectrum.git',JOHN_REV,'swoboda')
