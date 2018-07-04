from sklearn import model_selection, preprocessing, metrics

import pandas, numpy, keras
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers, initializers

# carrega o conjunto de dados
data = open('Dataset teste', encoding="utf8").read()

classes, textos = [], []
for i, line in enumerate(data.split("\n")):
    content = line.split()
    classes.append(content[0])
    textos.append(content[1])

# crie um dataframe usando textos e lables
trainDF = pandas.DataFrame()
trainDF['texto'] = textos
trainDF['classe'] = classes

# dividir o conjunto de dados em conjuntos de dados de treinamento e validação
train_x, valid_x, train_y, valid_y = model_selection.train_test_split( trainDF['texto'], trainDF['classe'])

# codifica o rótulo da variável de destino
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

# carregar os vetores de incorporação de palavras pré-treinados
embeddings_index = {}
for i, line in enumerate(open('Dataset Treino', encoding="utf8")):
    values = line.split()
    embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

# crie um tokenizer
token = text.Tokenizer()
token.fit_on_texts(trainDF['texto'])
word_index = token.word_index

# converta texto em sequência de tokens e guarde-as para garantir vetores de comprimento igual
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

# criar mapeamento de incorporação de tokens
embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):

    # ajuste o conjunto de dados de treinamento no classificador
    classifier.fit(feature_vector_train, label, epochs=5)

    # predizer os rótulos no conjunto de dados de validação
    classifier.predict(feature_vector_valid)



def create_cnn():
    # Adicione uma camada de entrada
    input_layer = layers.Input((70, ))

    # Adicione a camada de incorporação de palavras
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Adicione a camada convolucional
    conv_layer = layers.Convolution1D(90, 3,  activation="relu")(embedding_layer)

    # Adicione a camada de pooling máximo, pega maior valor resltande do mapa de  ativação.

    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Adicione as camadas de saída
    # camada totalmente conectada para normalização dos dados.
    output_layer1 = layers.Dropout(0.7)(pooling_layer)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile o modelo
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adamax(), loss='binary_crossentropy', metrics=['accuracy'])

    return model

classifier = create_cnn()
classifier.summary()

train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
