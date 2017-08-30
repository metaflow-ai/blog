# ML notes: Why the log-likelihood ?

## TL, DR!
### EN
Here is my first note on the machine learning subject about the reasons behind the use of the log-likelihood. I work out the equations from the definition of the likelihood to the log-likelihood removing any untold assumptions between each line of calculus.
Bonus: I add some remarks about SGD and MAP.

### FR
Voici ma première note sur le Machine Learning à propos des raisons d'usages du log-likelihood. Je refais le chemin de pensé qui mène de la définition du likelihood aux raisons de l'utilisation du log-likelihood en supprimant tout non dis pour chaque ligne de calcul.
En bonus, je fais quelques remarques sur les raisons du SGD et du MAP.


## Latex
```
\begin{align}
\underset{\theta}{\text{ argmax }} L(\theta, m | \mathcal{D}) &= \underset{\theta}{\text{ argmax }} p(\mathcal{D}|\theta, m) \\
&= \underset{\theta}{\text{ argmax }} p_{D_1, \ldots, D_n}(d_1, \ldots, d_n | \theta,m) \\ 
&\stackrel{\text{independence}}{=} \underset{\theta}{\text{ argmax }}  \prod_i p_{D_i}(d_i | \theta, m) \\
&\stackrel{\text{identically distributed}}{=} \underset{\theta}{\text{ argmax }} \prod_i p(d_i | \theta, m) \\
&= \underset{\theta}{\text{ argmax }} \sum_i \log(p(d_i | \theta, m)) \\
&= \underset{\theta}{\text{ argmax }} \frac{1}{n} \sum_i \log(p(d_i | \theta, m)) \\
&= \underset{\theta}{\text{ argmax }} E_{d \sim U_{\mathcal{D}}}[log(p(d|\theta,m))]
\end{align}
```

```
P(\mathcal{D}|\theta,m) = P(\mathcal{D}|\theta,\eta_1, \ldots, \eta_K, f_m)  
```