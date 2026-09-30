import pandas as pd

data = pd.read_csv("/Users/enriquegomez/Desktop/datos_.csv")

from pycopula_ml.copulas import FrankCopula, GaussianCopula
from pycopula_ml.estimation import fit_copula_mle

from time import time


t0 = time()
result = fit_copula_mle(
    GaussianCopula,
    data.u,
    data.v,
    bounds = (
        (-1, 1)
    )
)
t1 = time()

print(result.theta)
print(result.log_likelihood)
print(f"Valor obtenido en {t1-t0} segundos")
