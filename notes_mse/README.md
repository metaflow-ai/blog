
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
\underset{\theta}{\text{ argmax }} L(\theta, m | \mathcal{D}) = \underset{\theta}{\text{ argmax }} \sum_{d_i \in \mathcal{D}} \log(p(d_i | \theta, m))
```

```
\begin{align}
p(d^{(i)} | \theta, m) &= p(x^{(i)}, y^{(i)} | \theta, m) \\
&= p(y^{(i)} | x^{(i)}, \theta, m) * p(x^{(i)} | \theta, m) \\
&= p(y^{(i)} | x^{(i)}, \theta, m) * p(x^{(i)}) \\
\underset{\theta}{\text{ argmax }} \sum_i \log(p(d^{(i)} | \theta, m)) &= \underset{\theta}{\text{ argmax }} \sum_i \log(p(y^{(i)} | x^{(i)}, \theta, m))
\end{align}
```

```
\begin{align}
y^{(i)} &= h(x^{(i)} \times \epsilon^{(i)}) + \varepsilon^{(i)} \\
&\stackrel{\text{approx.}}{\approx} h_{\theta, m}(x^{(i)} \times \epsilon^{(i)}) + \varepsilon^{(i)} \\
&\stackrel{\text{no mul. noise}}{=} h_{\theta, m}(x^{(i)}) + \varepsilon^{(i)} \\
\end{align}
```

```
\varepsilon \sim \mathcal{N}(0, \sigma^2) &\Rightarrow p(\varepsilon^{(i)}) = \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(\varepsilon^{(i)})^2}{2\sigma^2}}
```

```
\begin{align}
&\text{We have } \varepsilon^{(i)} = y^{(i)} - h_{\theta, m}(x^{(i)}) \text{ and } \varepsilon \sim \mathcal{N}(0, \sigma^2) \\
&\Rightarrow p(y^{(i)} | x^{(i)}, \theta, m) = \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(y^{(i)} - h_{\theta, m}(x^{(i)}))^2}{2\sigma^2}} \\
&\Rightarrow log(p(y^{(i)} | x^{(i)}, \theta, m)) =  \log\frac{1}{\sqrt{2\pi}\sigma} - \frac{1}{\sigma^2} \frac{1}{2} (y^{(i)} - h_{\theta, m}(x^{(i)}))^2 \\
&\Rightarrow \underset{\theta}{\text{ argmax }} \sum_i \log(p(d^{(i)} | \theta, m)) = \underset{\theta}{\text{ argmin }} \frac{1}{2}\sum_i^N(y^{(i)} - h_{\theta, m}(x^{(i)}))^2
\end{align}
```


```
p(y^{(i)}|x^{(i)}, \theta, m) \neq p(y^{(i)}|x^{(i)}; \theta, m)
```
