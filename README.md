# ITS
Código desenvolvido durante projeto de graduação na UnB

O objetivo do projeto é segmentar e identificar (espécie) cada árvore de uma dada imagem.

COMO USAR:
 - Inicialmente é necessário treinar uma knn (uma pré treinada está disponível na pasta "pre_feitos").
    - Para o treinamento, use a classe knn_train, ela permitirá o treino iterativo.
    - O treinamento é baseado em separar o que é de interesse (árvores) do que não é de interesse (não árvore).
 - Para continuar é preciso usar o segmentador, ele criará um arquivo de segmentos que permite o treinamento de uma métrica.
 - Para conseguir executar a segmentação é preciso ter um arquivo de métrica e um arquivo de segmentos da imagem desejada. Para criar a métrica basta criar em "treinar métrica" para abrir a janela onde é preciso juntar segmentos que pertencem a mesma árvore.
     - Este treinamento é baseado em selecionar segmentos parecidos (mesma árvore) e selecionar a opção de juntar.
 - Para executar a segmentação basta usar a opção "contador".
 - Por fim pode-se selecionar a opção "classificador", que classifica as árvores encontradas em espécies.

   
Obs. 1: O modelo knn fica no local das imagens e é selecionado pelo usuário.

Obs. 2: O modelo knn pode ser agilizado usando LUT (look up table), então ao selecionar a knn para uso o arquivo de LUT será criado. No lugar de selecionar a knn, pode-se selecionar a LUT.
