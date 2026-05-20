from __future__ import annotations

###     Global variables    ###
#   USE_PYFFTW : True   - > use pyfftw
#                False  - > use numpy fft
USE_PYFFTW = False
#   USE_FREQUENCY_DOMAIN_RAMAN : 
#   True   - > calculate Raman respose in frequency domain (older)
#   False  - > calculate Raman reponse in time domain (Modern version)
USE_FREQUENCY_DOMAIN_RAMAN = False
#   USE_SIMPLE_RAMAN : 
#   True   - > use classic (Agarwal 1989) sin(t/t1)exp(-t/t2) response
#   False  - > use more modern three-time version (Lin & Agarwal 2006)
PRE_FFTSHIFT = True