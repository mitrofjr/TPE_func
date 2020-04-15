# TPE_func

A tool fo DFT functional fitting. More information: https://pubs.acs.org/doi/10.1021/acs.jpca.9b09093 , please cite if using:

Artem Mitrofanov, Vadim Korolev, Nikolai Andreadi, Vladimir Petrov, and Stepan Kalmykov. A simple automatized tool for exchange-correlation functional fitting. The journal of physical chemistry. A, 124(13):2700–2707, 2020.

To start functional fitting:

python run.py DIPCS10 -i 100 -b 10

where

DIPCS10 - name of folder with GMTKN55 (www.chemie.uni-bonn.de/pctc/mulliken-center/software/GMTKN/gmtkn55) subset

i - number of iterations

b - number of reactions in a batch


Included basic functionals from libxc (www.gitlab.com/libxc/libxc)

PySCF code (www.github.com/pyscf/pyscf) used as a basic QC calculator

Analisys model includes SHAP approach from (www.github.com/slundberg/shap)
