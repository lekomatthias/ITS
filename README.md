# ITS
Código desenvolvido durante projeto de graduação na UnB

O objetivo do projeto é segmentar e identificar (espécie) cada árvore de uma dada imagem.

COMO USAR:
 - Inicialmente é necessário treinar uma knn (uma pré treinada está disponível na pasta "pre_feitos").
    - Para o treinamento, use a classe knn_train, ela permitirá o treino iterativo.
    - O treinamento é baseado em separar o que é de interesse (árvores) do que não é de interesse (não árvore).
 - Para continuar basta usar a função "Type_visualization" da classe "ClusteringClassifier".
 - Para conseguir executar a segmentação é preciso usar a função Train, que abre uma janela para criar uma métrica de comparação entre superpixels.
     - Este treinamento é baseado em selecionar segmentos parecidos (mesma árvore) e selecionar a opção de juntar.

Obs.: A estrutura a partir de onde a imagem base está deve ter as pastas:
 - classificadas
 - segmentos
 - mascaras
 - metricas
   
O modelo knn fica no local das imagens e é selecionado pelo usuário.
