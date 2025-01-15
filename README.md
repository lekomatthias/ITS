# ITS
Código desenvolvido durante projeto de graduação na UnB

O objetivo do projeto é segmentar e identificar (espécie) cada árvore de uma dada imagem.

COMO USAR:
 - Primeiro é preciso treinar uma KD-tree executando o arquivo knn_treino.py.
 - Segundo é preciso usar o segmentador para obter um arquivo de segmentos, usando knn_slic_metric.py. Aqui é preciso abrir o arquivo para mudar a variável new_segments para inicializar como True.
 - Terceiro ainda usando o arquivo knn_slic_metric.py, mude new_segments para False, e train para true, e new_model para True, aqui selecionando o modelo de kd-tree, a imagem, e os segmentos criados, será permitido selecionar superpixels que pertencem à mesma árvore, é preciso selecionar todas as árvores que estão divididas separadamente para treinar a matriz métrica necessária para o próximo passo.
 - Quarto, ainda com knn_slic_metric.py, mude train para false. O programa pegará todos os segmentos vizinhos e usará a métrica treinada para tentar juntar árvores que são mais próximas tanto em posição quanto em cor.

Obs.: para selecionar os arquivos é importante ler o nome da janela aberta, ela fala qual é o arquivo esperado para cada seleção.
