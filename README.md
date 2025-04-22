# ITS
Código desenvolvido durante projeto de graduação na UnB

O objetivo do projeto é segmentar e identificar (espécie) cada árvore de uma dada imagem.

COMO USAR:
 - Executar o arquivo main.py
 - Ele pede para selecionar uma imagem, ela será usada para fazer todas as segmentações.
 - Para conseguir executar o a segmentação é preciso usar a função Train, que abre uma janela para treinar uma knn.
    - O treinamento é baseado em separar o que é de interesse (árvores) do que não é de interesse (não árvore).

Obs.: A estrutura a partir de onde a imagem base está deve ter as pastas:
 - classificadas
 - segmentos
 - mascaras
 - metricas
   
O modelo knn fica no local das imagens e é selecionado pelo usuário.
