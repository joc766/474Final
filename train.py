
import numpy as np 
import sys

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.saved_model import SaveOptions

from peg_game import PeggingGame
from test_policies import create_agent


from simulate import simulate

N_SIMULATIONS = 12000

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

    # Simulate the games
    x_all = []
    y_all = []

    for i in range(N_SIMULATIONS):
        if i % 100 == 0:
            print("Simulation", i, file=sys.stderr)
        game = PeggingGame(4)
        agent1 = create_agent("mcts")
        agent2 = create_agent("minimax")
        sim_results = simulate(game, agent1, agent2)
        x_all.append(sim_results[0])
        y_all.append(sim_results[1])
    
    print("SIMULATION COMPLETE", file=sys.stderr)

    # one-hot encode the x data
    x_encoded = np.array([create_encodings(cards) for cards in x_all])

    # one-hot encode the y data
    y_encoded = []
    for y in y_all:
        encoding = np.zeros(21)
        encoding[y+10] = 1
        y_encoded.append(encoding)
    y_encoded = np.array(y_encoded)
    
    # split into training data and test data
    test_size = 100

    x_test = x_encoded[:test_size] 
    y_test = y_encoded[:test_size]

    x_train = x_encoded[test_size:]
    y_train = y_encoded[test_size:]

    model = Sequential()
    model.add(Flatten(input_shape=(4, 17)))
    model.add(Dense(64, activation='relu', input_shape=(68,)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x=x_train, y=y_train, batch_size=25, epochs=100)

    # Save the model
    model.save("model.h5", save_format="tf")

    # Evaluate the model on the testing data
    predictions = model.predict(x_test)
    correct = y_test
    for p, c in zip(predictions, correct):
        print(f"PREDICTION: {', '.join('{:.2f}'.format(np.round(x,2)) for x in p)}")
        print(f"CORRECT:    {', '.join('{:.2f}'.format(np.round(x,2)) for x in c)}\n")

if __name__ == "__main__":
    main()
