# Fake_News_Ckassifiers
TCC de analise de algoritimos de machine learning e NLP para detecção de fakeNews

Ambiente python 3.9

Como utilizar o projeto:
Apos clonar o repositorio é necessario baixar o vetor de dimensão do glove no seguinte link : http://www.nilc.icmc.usp.br/nilc/index.php/repositorio-de-word-embeddings-do-nilc

Selecionar a opção de 50 dimensões

![image](https://user-images.githubusercontent.com/44040667/121757151-f9fdfa80-caf2-11eb-8a08-e290433c4ac9.png)

Apos o download do vocabulario do glove, no arquivo classifiers.py, editar as seguintes variaveis

path: inserir o caminho do arquivo que esta na pasta input/processadas.csv 

GLOVE_DIR: inserir o caminho onde se encontra o arquivo do vetor de dimensão do glove baixado anteriomente

![image](https://user-images.githubusercontent.com/44040667/121757264-611baf00-caf3-11eb-889c-b6c13e23896c.png)

Para execução dos algoritimos utilizar a classe main para chamar os metodos exemplificados na propria main

![image](https://user-images.githubusercontent.com/44040667/121757779-100cba80-caf5-11eb-91c0-2bd4cea0e47b.png)

