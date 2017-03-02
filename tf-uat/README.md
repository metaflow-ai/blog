# TensorFlow howto: a universal approximator inside a neural net

## TL, DR! 
### EN
- I implement a first universal approximator with TesnorFlow and i train it on a sinus function (I show that it actually works) 
- I use it inside a bigger neural networks to classify the MNIST dataset 
- I display the learnt activation functions 
- I show that whatever the learnt activation function is, i get consistently the accuracy 0.98 on the test set 
- bonus: all the code is open-source

### FR
- J'implémente une fonction d'approximation universelle avec Tensorflow et je l'entraine sur une fonction Sinus (je montre que ça marche :) )
- Je l'intègre à l'intérieur d'un réseau de neurones plus larges pour classifier le dataset MNIST
- Je présentes les fonctions d'activation ainsi apprises
- Je montre que quelquesoit la fonction d'activation obtenu, j'obtiens toujours la même "accuracy" de 0.98 sur les données de test
- En bonus, tous le code est open-source et commenté