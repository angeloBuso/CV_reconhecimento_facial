# Módulo para (i) Carregar o Dataset, (ii) pré-processar a imagem e (iii) splitar os dados em treino e teste

# Bibliotecas para uso no projeto
import numpy as np
import random
import cv2
from sklearn.datasets._base import Bunch
from imutils import paths
from scipy import io

def load_caltech_faces(datasetPath, min_faces=10, face_size=(47, 62), equal_samples=True, test_size=0.33, seed=42, flatten=False):
	"""
	:param datasetPath: indica o diretório que encontra a (i.) imagem formato jpg e (ii.) etiquetagem - pessoa que se trata
	:param min_faces: indica a quantidade mínima de exemplos por pessoa para compor o dataSet do modelo ->
	[Trade-off: quanto menor, maior será a quantidade de pessoas a serem tratadas, logo aumenta processamento]
	:param face_size: indica o tamanho do redimensionamento da face da imagem
	:param equal_samples: indica se seguirá apenas quantidade igual ou superior de imagens para o modelo
	:param test_size: indica qual percentual será realizado o split dos dados
	:param seed: indica uma semente de aleatoriedade para replicar mesmos resultados
	:param flatten: coloca a face na forma UNIdimensional, ou seja, "achata" o rosta em uma linha
	:return: dados de treino e teste, com as faces processadas para repasse ao modelo de ML
	"""
	# Parte 1: Obtém os caminhos das imagens associadas às faces e carrega os dados para ROI - Region of Interest
	imagePaths = sorted(list(paths.list_images(datasetPath)))
	bbData = io.loadmat("{}/ImageData.mat".format(datasetPath))
	bbData = bbData["SubDir_Data"].T

	# Parte 2: Inicializar variáveis [dados e labels], define seed [para replicação de exemplos]
	random.seed(seed)
	data = []
	labels = []

	# Parte 3: Pré processamento da imagens [loop para cada imagem: (a) leio o path da imagem, (b) coverto para cinza,
	# (c) obtenho o ROI dessa imagem - região da face -, (d) aplico o ROI na imagem já em escala de cinza, obtendo a face,
	# (e) redimensiona as faces para um mesmo tamanho - tamanho canonico, (f) converte para um objeto unidimensional
	# as imagens dos rostos ,(g) atualiza as listas dos dados e label criados na parte 2 e (h) converte a matriz de dados
	# e a lista de etiquetas para uma matriz NumPy.
	for (i, imagePath) in enumerate(imagePaths):
		
		# (a) à (e)
		image = cv2.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		k = int(imagePath[imagePath.rfind("_") + 1:][:4]) - 1
		(xBL, yBL, xTL, yTL, xTR, yTR, xBR, yBR) = bbData[k].astype("int")
		face = gray[yTL:yBR, xTL:xBR]
		face = cv2.resize(face, face_size)

		# (f)
		if flatten:
			face = face.flatten()

		# (g)
		data.append(face)
		labels.append(imagePath.split("\\")[-2])

	# (h)
	data = np.array(data)
	labels = np.array(labels)

	# Parte 4: (a) permite usar apenas as imagens com faces que contém uma quantidade igual ou superior a quantidade indicada
	# e (b) garante a aleatoriedade nas escolhas de imagens
	if equal_samples:
		# Inicializa a lista de índices 
		sampledIdxs = []

		# Loop pelos labels únicos
		for label in np.unique(labels):
			# Obtém os índices na matriz de etiquetas onde os rótulos são iguais ao rótulo atual
			labelIdxs = np.where(labels == label)[0]

			# Prossegue apenas se o número necessário de rostos mínimos puderem ser atendidos
			if len(labelIdxs) >= min_faces:
				# Aleatoriamente gera os índices para o rótulo atual, mantendo apenas o valor mínimo fornecido, e atualiza a lista de índices amostrados
				labelIdxs = random.sample(list(labelIdxs), min_faces)
				sampledIdxs.extend(labelIdxs)

		# Usa os índices amostrados para selecionar os pontos e os rótulos de dados apropriados
		random.shuffle(sampledIdxs)
		data = data[sampledIdxs]
		labels = labels[sampledIdxs]

	# Parte 5: Split dos dados; indica o índice de divisão de treinamento e teste
	idxs = list(range(0, len(data)))
	random.shuffle(idxs)
	split = int(len(idxs) * (1.0 - test_size))

	(trainData, testData) = (data[:split], data[split:])
	(trainLabels, testLabels) = (labels[:split], labels[split:])

	# Cria os grupos de treinamento e teste
	training = Bunch(name="training", data=trainData, target=trainLabels)
	testing = Bunch(name="testing", data=testData, target=testLabels)

	# Devolve uma tupla do treinamento
	return (training, testing, labels)

	