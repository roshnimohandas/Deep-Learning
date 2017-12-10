"# Deep-Learning" 

# This is a tutorial on ANN with two hidden layers. 

##ANN Steps
1) Randomly initialise weights (Close to 0, but not exactly 0)
2) Input first observation in the dataset in the input layer , each feature in one input layer 
3) Forward propagation : From left to right ,the neurons are actiavated in a way that the impact of each neurons activity is limited by the weights. Propagate the activations until getting the 
predicted results y 
4) Compare the predicted resul and actual. Measure the error 
5) Back propagation from right to left . update the weights 
6) Repeat steps 1 to 5 and update the weights after each obs( Reinforcement learning) or repeat steps 1 to 5 and update the weights after a batch of obs(Batch learning)
7) When a whole training set is passed through ANN, thats an epoch, repeat epochs