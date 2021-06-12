# Fake_News_Classifiers
TCC de analise de algoritmos de machine learning e NLP para detecção de fakeNews

Ambiente python 3.9


Como utilizar o projeto:

Após clonar o repositório é necessário baixar o vetor de dimensão do glove no seguinte link: 
http://www.nilc.icmc.usp.br/nilc/index.php/repositorio-de-word-embeddings-do-nilc

Selecionar a opção de 50 dimensões

![image](https://user-images.githubusercontent.com/44040667/121758121-7e9e4800-caf6-11eb-9517-b115fd70c473.png)

Após o download do vocabulário do glove, no arquivo classifiers.py, editar as seguintes variáveis

path: inserir o caminho do arquivo que se encontra na pasta input/processadas.csv 

GLOVE_DIR: inserir o caminho onde se encontra o arquivo do vetor de dimensão do glove baixado anteriormente

```python
path = 'C:/Users/gabis/Documents/tcc/Fake_News_Ckassifiers/input/processadas.csv'
folder = 'C:/Users/gabis/Documents/tcc/Fake_News_Ckassifiers/input/dataToProcess'
GLOVE_DIR = 'C:/Users/gabis/PycharmProjects/Glove/input/glove_s50.txt'
```

Para execução dos algoritmos utilizar a classe main para chamar os métodos exemplificados na própria main
```python
#metodos svm
classifiers.svm()
classifiers.svmCross()
classifiers.svm_all_validations()
#metodos naive bayes
classifiers.naive_bayes()
classifiers.naive_bayesCross()
classifiers.naive_bayes_all_validations()
#metodo glove
classifiers.glove()
```
