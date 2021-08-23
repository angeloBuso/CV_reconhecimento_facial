# Reconhecimento Facial com a técnica Eigenfaces

# Bibliotecas para uso no projeto
import numpy as np
import argparse
import imutils
import cv2
from datasets import load_caltech_faces
from resultsmontage import ResultsMontage
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from skimage import exposure

# Argumentos: criação de argumentos que automatiza o script
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Caminho para o dataset da Caltech")
ap.add_argument("-n", "--num-components", type=int, default=150, help="Numero de Componentes Principais")
ap.add_argument("-s", "--sample-size", type=int, default=10, help="Numero de Amostras")
ap.add_argument("-v", "--visualize", type=int, default=-1, help="Se os componentes PCA devem ser visualizados")
args = vars(ap.parse_args())

# Parte 1: Carga dos dados: chamada ao módulo "datasets" que possui uma função que realiza a (i) carga do Dataset,
# (ii) pré-processamento da imagem e (iii) split em treino e teste.

print("Carga, pré-processamento e split das imagens...")
(training, testing, names) = load_caltech_faces(args["dataset"], min_faces=21, flatten=True, test_size=0.25)

# Parte 2: Computa a representação PCA (eigenfaces) dos dados e, em seguida, projeta os dados de treinamento no subespaço de autodefinições (Eigenspace)
# EigenSpace: (espaço de Imagens) é um vetor de matrizes e cada elemento do vetor é uma matriz que representa uma imagem.
print("Criando Eigenfaces...")
pca = PCA(n_components=args["num_components"], whiten=True)
trainData = pca.fit_transform(training.data)

# Parte 3: Visualização dos componentes do PCA
if args["visualize"] > 0:
	# Inicializa a montagem para os componentes
	montage = ResultsMontage((62, 47), 4, 16)

	# Loop sobre os primeiros 16 componentes individuais
	for (i, component) in enumerate(pca.components_[:16]):

		# Reshape do componente para uma matriz 2D e, em seguida, converte o tipo de dados em um inteiro de 8 bits 
		# para que ele possa ser exibido com o OpenCV
		component = component.reshape((62, 47))
		component = exposure.rescale_intensity(component, out_range=(0, 255)).astype("uint8")
		component = np.dstack([component] * 3)
		montage.addResult(component)

	# Mostra a média e as visualizações dos componentes principais 
	# Mostra a imagem média
	mean = pca.mean_.reshape((62, 47))
	mean = exposure.rescale_intensity(mean, out_range=(0, 255)).astype("uint8")
	cv2.imshow("Media", mean)
	cv2.imshow("AutoVetores das faces - PCA", montage.montage)
	cv2.waitKey(0)

# Parte 4: Treinamendo do modelo, com os compomentes das imagens!
print("Treinando o classificador...")
model = SVC(kernel="rbf", C=10.0, gamma=0.001, random_state=84)
model.fit(trainData, training.target)

# Parte 5: Avaliação do modelo
print("Avaliando o modelo...")
predictions = model.predict(pca.transform(testing.data))
print(classification_report(testing.target, predictions))

# Parte 6: Realizando Previsão

# Obtenção de algumas imagens aleatórias
for i in np.random.randint(0, high=len(testing.data), size=(args["sample_size"],)):
	
	# Obtém o rosto e classifica (o classificador espera receber um PCA da imagem e não uma matriz)
	face = testing.data[i].reshape((62, 47)).astype("uint8")
	prediction = model.predict(pca.transform(testing.data[i].reshape(1, -1)))

	# Visualização da previsão
	print("Previsao do Modelo: {}, Quem realmente é: {}".format(prediction[0], testing.target[i]))
	face = imutils.resize(face, width=face.shape[1] * 2, inter=cv2.INTER_CUBIC)
	cv2.imshow("Face", face)
	cv2.waitKey(0)