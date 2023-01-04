## Description:

I have trained a neural network using tensorflow to predict the outcome of the pegging phase of cribbage
based on the four cards that a player has in their hand. The neural network, given a list of the cards in a player's hand, produces a probability distribution of various different outcomes of the pegging phase. Originally, I wanted to produce a numeric prediction for the number of points scored. However, I found that it would be more 
useful to use the probability distribution of each move when it came to the application of the 
neural network to the keep phase of cribbage. That way, I could get the expected reward for a 
given action the same way that we did for Markov-Decision processes. So, to produce an 
expected value, I summed the probability of an outcome multiplied by the reward for that outcome for each possible 
combination of four cards from the six that were dealt to the player. 


## How to use:

Makefile includes instructions to make two executeables: CreateModel and TestModel. 
- Run `./CreateModel [n_training_simulations]` to create the model with the desired number of test cases
    - To create the CreateModel executable run `make CreateModel`
    - Run the model with > 10 n_training_simulations as I have it set to only have 10 training simulations
    - This takes a long time to make a good model. Less than a few thousand runs makes pretty bad predictions.
      I got the best results from 100,000+ simulations. My final model used 1,000,000 simulations and took 
      several hours to train
    - The new model will be located in `new_model.h5`.
- Run `./TestModel [n_test_games]` to test the policy in my_policy.py that uses the model to determine
  which cards to keep in the keep() function. Running `./TestModel` is equivalent to running `./TestCribbage` in 
  pset1
    - To create the TestModel executable run `make TestModel`
    - As pypy3 does not work with numpy or tensorflow, this takes a very long time to run enough simulations.
      I have results from my simulations located in metrics/test_new_agent.txt
    - My default is to use the model that I spent a longer time training. 
      If the default model (`good_greedy_model.h5`) does not work, change the file in load_model 
      on line 12 in my_policy.py to the `new_model.h5` that was trained using the CreateModel 
      executable. Note that unless trained for a similar amount of time this will perform poorly.


## Steps:

I adapted the code from pset 1 and pset 4 to train the model. To simulate games for inputs to the model, I adapted the code from pset4/test_mcts.py and the compare_policies() function to simulate the pegging phase specifically. Then, I took my heuristic policy's peg() function and adapted it to be like the mcts_policy() function that returns a policy function to be used to simulate the pegging phase. I then did the same for the greedy policy from pset 1. I one-hot encoded the list of four cards given to player 1 (my heuristic agent) for the inputs to the model and also one-hot encoded the resulting score of the simulation to be a series 
of bits representing a score from -31 to 31 points (where 31 is the absolute maximum result of the pegging phase) for my outputs.
I trained the model over 1 million simulations of the game to fine-tune it. 


## Results:

The results of the model's predictive ability for the pegging phase only can be seen in both the `training_results.txt` file and the `pegging_accuracy.out` file. The model always picked an outcome that was possible, but rarely did the model predict an outcome with greater than 20% probability. Therefore, I conclude that the stochastic nature of cribbage and the fact that it is an imperfect information game limits the ability of the neural network to make useful predictions. 

While the model did create a reasonable prediction for the number of points that would be scored in pegging,
this information was not particularly useful when it came to deciding which cards to discard. At least, choosing 
the hand that had the highest probability of scoring well during the pegging phase did not make much of a 
difference. The results of the new and old agents in 1000 games of cribbage can be seen in 
`metrics/test_new_agent.txt` and `metrics/test_old_agent.txt` respectively. 

However, I chose not to give significant weight to the predicted score of the pegging phase as most of the time,
the number of points expected from the hand itself (calculated by the `score()` function) was greater than the
number of points expected from the hand itself or the crib. Initial tests discarding the scores of the crib and/or
the hand resulted in worse performance. The results of the policy that relied only on the model's pegging score prediction are in `metrics/new_policy_results.out` (0.19 avg points) and the results of the policy that relied on score of the hand and the pegging prediction are in `metrics/old_policy_results.out` (0.212 avg points). (Note that the difference between this comparison and the previous one is that both of these policies use the model, with different weights on its predictions for selection, while one of the policies in the previous paragraph uses the model and one of the policies does not use the model. )
