
import numpy as np 

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

from peg_game import PeggingGame
from test_policies import create_agent


from simulate import simulate

N_SIMULATIONS = 100

# Create a list of all possible ranks and suits
RANKS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
SUITS = ['S', 'D', 'C', 'H']

# Function to convert a list of cards into a feature tensor
# TODO change to 52 bit encoding
def create_encodings(cards):
    # Loop through each card and create the one-hot encoding
    input_data = None
    for card in cards:
        rank, suit = card
        rank_encoding = np.zeros(13)
        rank_encoding[rank-1] = 1
        suit_encoding = np.zeros(4)
        if suit == 'S':
            suit_encoding[0] = 1
        elif suit == 'D':
            suit_encoding[1] = 1
        elif suit == 'C':
            suit_encoding[2] = 1
        else:
            suit_encoding[3] = 1
        encoding = np.concatenate((rank_encoding, suit_encoding))
        if input_data is not None:
            input_data = np.vstack((input_data, encoding))
        else:
            input_data = encoding
    return input_data

def main():
    x_all = []
    y_all = []

    for i in range(N_SIMULATIONS):
        game = PeggingGame(4)
        agent1 = create_agent("mcts")
        agent2 = create_agent("minimax")
        sim_results = simulate(game, agent1, agent2)
        x_all.append(sim_results[0])
        y_all.append(sim_results[1])

    # Convert the input data into feature tensors
    x_encoded = np.array([create_encodings(cards) for cards in x_all])

    # find the max and min of the y data
    max_y = max(y_all)
    min_y = min(y_all)
    size_range = max_y - min_y
    # one-hot encode the y data
    y_encoded = []
    for y in y_all:
        encoding = np.zeros(size_range)
        encoding[y - min_y] = 1
        y_encoded.append(encoding)
    y_encoded = np.array(y_encoded)
    print(y_encoded)
    
    # split into training data and test data
    test_size = int(len(x_all) / 5)
    train_size = len(x_all) - test_size

    x_test = x_encoded[:test_size] # TODO look up why not matrix
    y_test = np.array(y_encoded[:test_size])

    x_train = x_encoded[test_size:]
    y_train = np.array(y_encoded[test_size:])
    print(x_train.shape)
    print(y_train.shape)
    print(len(x_train))
    print(len(y_train))

    model = Sequential()
    model.add(Dense(10, input_dim=x_train.shape[1]))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation=None))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    model.fit(x=x_train, y=y_train, batch_size=32, epochs=10)

    # Evaluate the model on the testing data
    predictions = model.predict(x_test)
    correct = y_test
    for p, c in zip(predictions, correct):
        print("PREDICTION: ", p)
        print("CORRECT: ", c)
    print(model.summary())

if __name__ == "__main__":
    main()
