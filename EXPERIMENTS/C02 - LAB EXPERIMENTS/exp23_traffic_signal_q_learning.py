import numpy as np

# Tiny traffic environment:
# State 0 = more cars on North-South road; State 1 = more cars on East-West road.
# Action 0 = green for North-South; Action 1 = green for East-West.
# Reward is negative waiting traffic after the action.
np.random.seed(42)
q_table = np.zeros((2, 2))
learning_rate = 0.2
discount = 0.9
epsilon = 0.2

def next_state_and_reward(state, action):
    # Reward favors giving green to the busier direction.
    if state == action:
        reward = 10
        new_state = np.random.choice([0, 1], p=[0.7, 0.3])
    else:
        reward = -5
        new_state = np.random.choice([0, 1], p=[0.4, 0.6])
    return new_state, reward

episodes = 2000
for _ in range(episodes):
    state = np.random.randint(0, 2)
    if np.random.rand() < epsilon:
        action = np.random.randint(0, 2)
    else:
        action = int(np.argmax(q_table[state]))

    new_state, reward = next_state_and_reward(state, action)
    q_table[state, action] += learning_rate * (
        reward + discount * np.max(q_table[new_state]) - q_table[state, action]
    )

print("Learned Q-table (rows=traffic state, columns=signal action):")
print(np.round(q_table, 2))
for state in [0, 1]:
    action = int(np.argmax(q_table[state]))
    signal = "North-South GREEN" if action == 0 else "East-West GREEN"
    print(f"State {state}: {signal}")

print("\nThis is a simplified simulation for demonstrating Q-learning, not a live traffic controller.")
