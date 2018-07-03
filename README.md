## Introdução
Esta é a implementação do artigo [Classificação de Linguagem Natual com Redes Convolucionais] ()

## Requerimentos
* python 3
* Keras
* Pandas
* Numpy
* Sckit
* scikit-learn

## Datasets
### Para Treinamento
* [wiki-news-300d-1M.vec.zip](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M.vec.zip)

### Para Teste
* [corpus](https://gist.github.com/kunalj101/ad1d9c58d338e20d09ff26bcc06c4235)


## Ajustando

Agora altere  os endereços de entrada dos datasets para os endereços correspondentes  em seu repositorio. sendo o primeiro  dataset de teste  e o segundo de treinamento.
<code>
* 1 - open('Dataset teste', encoding="utf8")
* 2 - open('Dataset Treino', encoding="utf8")
</code>

## Teste e Resultado da CNN

Após  ter todas as dependencias e configurado os Datasets podemos executar a CNN de dentro do repositorio da seguinte forma:

```
python cnn.py
```

A execução carregara os Datasets e formara a rede, em seguida apresentara  os porcedimentos de testes em 5 epocas de execução.

![title](configuração.jpg)

Com esta configuração da CNN obterá resultados na taxa de acerto da rede de 75% confome visto na figura seguinte.


![title](resultado.jpg)