# Reinforcement Learning for Financial Trading Models

The files in this folder contains the definition of the agent that will be used in the various testing environments in the root folder such as `DRL_BOC+CNOOC.ipynb`, `DRL_BOC.ipynb`, and `DRL_AIA+HSE.ipynb`. The codebase uses the PyTorch framework.

`agent.py` defines the abstract base class that the DRLAgent in `drl_agent.py` will inherit from. 
Thus, it has the required functions trade, train, load_model, and save_model. 

`drl_agent.py` is the class used in the environment. A brief explanation of the code is as follows and a more detailed explanation can be found in the comments in the file itself:

We define the Actor class in `drl_agent.py` as the neural network architecture that is responsible for producing the actionable outputs after receiving inputs from the environment as well as the one that calculates the rewards that it gets. Since this Actor class defines the neural network, it must contain the function 'forward'. In PyTorch, this means that the inputs are passed through the network to produce the action and the reward. PyTorch uses AutoGrad which means that it will adjust the weights of the network after each iteration automatically without it having to be specified.

DRLAgent is our Agent and contains an instance of the Actor class that we have defined. Our DRLAgent inherits from the Agent class defined in `agent.py`. Our Agent instantiates this actor when it is intialized and has several important functions, namely  trade, train, load_model, and save_model.

DRLAgent has two trade functions, _trade and trade. The difference between the two functions is that '_trade' temporarily sets all the requires_grad flag to false and that means this is particularly useful when evaluating the model and to prevent tracking history and no weight update is done. The 'trade' function will then call this '_trade' function to get an action (without a weight update).

The train function simply performs training on the input dataset by calling the forward function defined by the Actor. load_model and save_model simply loads and saves.

The DRLAgent is then called in the environment notebooks. :)



