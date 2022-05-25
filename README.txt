Modelul asta l-am inspirat de aici:
https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/ICPRAM_CHESS_DNN_2018.pdf

A three hidden layer deep perceptron with 2048 hidden units per layer has been used.
Each layer is activated by the Elu activation function and the SGD training parameters have been initialized as follows:
η = 0.001; ε = 1e − 0.8 in combination with a Nesterov Momentum of 0.7. In addition to that Batch Normalization between
all the hidden layers and Minibatches of 256 samples have been used.

In fisierul only_rated_new_rating.csv am datele, unde am pe fiecare linie 773 de biti ce reprezinta tabla in acea
pozitie si ultimul element de pe linie reprezinta rating-ul dat de stockfish pentru acea pozitie, normalizat la
(1,0)

Imi pot face fisiere cu date mult mai mari, dar momentan testez pe asta.