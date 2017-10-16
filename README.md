# HiddenMarkovModel
An simple implementation of Hidden Markov Model

## how to use
The model can be created by calling 
```
hm = HiddenMarkov(n_hidden, n_visible)
```

After being created, the model can be trained by passing a sequence of observations formatted as python list to the fit method.

```
hm.fit(list_of_observations)
```

Depend on the number of hiddens and visible states, the training time may vary a lot as the complexity is proportional to the square of the number of hidden states and to the number of visible states. After being fitted, the model can be used to the predict the next observations given the list of observations:

```
prediction = hm.predict(list_of_observations)
```
