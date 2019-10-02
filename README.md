# TPE_func

To start functional fitting:
python run.py DIPCS10 -i 100 -b 10

where
DIPCS10 - name of folder with GMTKN55 (www.chemie.uni-bonn.de/pctc/mulliken-center/software/GMTKN/gmtkn55) subset
i - number of iterations
b - number of reactions in a batch

Included basic functionals from libxc (gitlab.com/libxc/libxc)

PySCF code (github.com/pyscf/pyscf) used as a basic QC calculator

Analisys model includes SHAP approach from (github.com/slundberg/shap)
