# Comandos ejecutados en la prueba de recuperación

```
[    1.1 s] git clone https://github.com/davidbrinez30/defiende-el-servidor-unsw-nb15.git proyecto
[    0.1 s] git checkout main
[    0.0 s] git rev-parse HEAD
[    6.7 s] ~\Documents\claude\defiende-el-servidor\defiende-servidor\.venv\Scripts\python.exe -m venv ~\recuperacion-defiende-20261024-144645\venv
[  207.5 s] ~\recuperacion-defiende-20261024-144645\venv\Scripts\python.exe -m pip install -r requirements.txt
[    2.9 s] ~\recuperacion-defiende-20261024-144645\venv\Scripts\python.exe -m pip freeze
[    9.0 s] ~\recuperacion-defiende-20261024-144645\venv\Scripts\python.exe -m dvc pull data/raw/UNSW_NB15_training-set.csv.dvc data/raw/UNSW_NB15_testing-set.csv.dvc prepare release_v1 release_v2
[   13.7 s] ~\recuperacion-defiende-20261024-144645\venv\Scripts\python.exe -m src.cli verificar
[    5.4 s] ~\recuperacion-defiende-20261024-144645\venv\Scripts\python.exe -c from src.predict import descargar; print(descargar('3c9aabf11b57387f55d07261458c574f241be34c', 'C:\\Users\\SANTIAGO_Y _SAMUEL\\recuperacion-defiende-20261024-144645\\hf\\v1'))
[    3.6 s] ~\recuperacion-defiende-20261024-144645\venv\Scripts\python.exe -m src.predict --release ~\recuperacion-defiende-20261024-144645\hf\v1 --input ~\recuperacion-defiende-20261024-144645\proyecto\evidencias\recuperacion\entradas_validacion.csv --output ~\recuperacion-defiende-20261024-144645\proyecto\evidencias\recuperacion\salida_despues_v1.csv
[    4.4 s] ~\recuperacion-defiende-20261024-144645\venv\Scripts\python.exe -c from src.predict import descargar; print(descargar('3639b0994cd1493723b8d8899f0e8311d9c798c3', 'C:\\Users\\SANTIAGO_Y _SAMUEL\\recuperacion-defiende-20261024-144645\\hf\\v2'))
[    2.9 s] ~\recuperacion-defiende-20261024-144645\venv\Scripts\python.exe -m src.predict --release ~\recuperacion-defiende-20261024-144645\hf\v2 --input ~\recuperacion-defiende-20261024-144645\proyecto\evidencias\recuperacion\entradas_validacion.csv --output ~\recuperacion-defiende-20261024-144645\proyecto\evidencias\recuperacion\salida_despues_v2.csv
```
