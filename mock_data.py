import numpy as np
import os

file = 'data/tf_data.npy'

# Create folder if not exists
os.makedirs('data', exist_ok=True)

train_data = []

# Number of samples
NUM_SAMPLES = 1000

for i in range(NUM_SAMPLES):
    # Fake image (100x100 RGB)
    screen = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # Fake steering (-1 to 1)
    steering_angle = np.round(np.random.uniform(-1, 1), 2)

    # Fake throttle (0 to 0.8 like your constraint)
    throttle = np.round(np.random.uniform(0, 0.8), 2)

    control = [steering_angle, throttle]

    train_data.append([screen, control])

# Save file
np.save(file, np.array(train_data, dtype=object))

print(f"Mock data created: {len(train_data)} samples")