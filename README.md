# Reconhecimento Facial

Replicar a habilidade humana de reconhecer pessoas nos computadores, exige uso de técnicas de pré-processamento de imagem. Neste projeto usando a técnica *Eigenface* extrai-se do conjunto de imagens os **autovetores** da matriz das imagens com propósito de identificar similaridade das faces.
O projeto é composto por 3 scripts (i) eigenfaces_face_recognition - script principal, (ii) datasets - script que carrega, processa e *splita* os dados e (iii) resultsmontage - script que imprime as matriz de covariância, o *eigenface*!

Execute o comando abaixo no terminal ou prompt de comando:

python eigenfaces_face_recognition.py --dataset <path do arquivo caltech_faces>
![Alt Text](https://github.com/angeloBuso/CV_reconhecimento_facial/blob/master/cv_deteccao_faces.gif)

  
  Referências:
  * https://www.lcg.ufrj.br/marroquim/courses/cos756/trabalhos/2013/abel-nascimento/abel-nascimento-report.pdf
  * https://en.wikipedia.org/wiki/Eigenface
  * https://www.cs.ucsb.edu/~mturk/Papers/mturk-CVPR91.pdf
  * https://arxiv.org/pdf/1705.02782.pdf
  * http://vision.stanford.edu/teaching/cs231a_autumn1112/lecture/lecture2_face_recognition_cs231a_marked.pdf
  * https://pt.khanacademy.org/math/linear-algebra/alternate-bases/eigen-everything/v/linear-algebra-introduction-to-eigenvalues-and-eigenvectors
