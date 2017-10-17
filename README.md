# Hidden Markov Model
An simple implementation of Hidden Markov Model

## How to use
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
## Theory
Please refer to "hidden_markov_model.pdf" for explanation of how the model is constructed.

## Performance
The following plots show how the model was used to reproduce the data that has been used to trained the model at different number of hidden states.

![Alt text](https://github.com/datduonguva/HiddenMarkovModel/blob/master/4-60.png?raw=true "4 hidden states")
![Alt text](https://github.com/datduonguva/HiddenMarkovModel/blob/master/6-60.png?raw=true "6 hidden states")
![Alt text](https://github.com/datduonguva/HiddenMarkovModel/blob/master/8-60.png?raw=true "8 hidden states")
![Alt text](https://github.com/datduonguva/HiddenMarkovModel/blob/master/12-60.png?raw=true "12 hidden states")
