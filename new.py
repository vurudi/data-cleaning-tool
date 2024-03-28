import pandas as pd
import numpy as np

# Define the size of the dataset
num_records = 1000

# Create a DataFrame with random values
data = {
    'ID': range(1, num_records + 1),
    'Name': ['User' + str(i) for i in range(1, num_records + 1)],
    'Age': np.random.choice([20, 25, 30, 35, 40, np.nan], size=num_records),  # Introduce missing values
    'Gender': np.random.choice(['Male', 'Female'], size=num_records),
    'Salary': np.random.randint(30000, 100000, size=num_records)
}

df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv('missing_values_dataset.csv', index=False)













import pandas as pd
import numpy as np

# Define the size of the dataset
num_records = 1000

# Create a DataFrame with random values
data = {
    'ID': range(1, num_records + 1),
    'Name': ['User' + str(i) for i in range(1, num_records + 1)],
    'Age': np.random.choice([20, 25, 30, 35, 40], size=num_records),
    'Gender': np.random.choice(['Male', 'Female'], size=num_records),
    'Salary': np.random.choice(['30000', '50000', '70000', '90000', 'NaN'], size=num_records)  # Introduce incompatible data
}

df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv('incompatible_data_dataset.csv', index=False)
