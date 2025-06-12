# /game_agent/dqn/model/dqn_model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from keras.optimizers import Adam

def build_dqn_model(state_size=4, action_size=5, learning_rate=0.001):
    """
    Construye un modelo DQN básico con Keras.
    """
    model = Sequential()
    model.add(Input(shape=(state_size,)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))  # Q-values
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    return model
