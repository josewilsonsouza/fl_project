# Info
* Comando que gera `estrutura.txt`: `tree /F /A > estrutura.txt` (no powershell)
* Optuna para o caso distribuído parece interessante
* Ao inserir os datasets VED e eVED no Hugginface eu apaguei o `;` que ficava no final (no dataset `eVED_180124_week.csv`), pois estava insconsistente

# ToDo
* testar no dados eVED e VED
* Incorporar o DCAINet
* Incorporar GRU
* Criar cenário com fluxo de dados

# Progress
- fiz um código que gera as rotas para um dado client e trip
- tenho um código que pega os dados que serão usados para treino e teste do hf
  e então salva em disco para que possa ser usado pelo Fleven
- alterei o `utils.py` e o `pyproject.toml` para acompanhar a simulção no eVED