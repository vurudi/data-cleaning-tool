import pandas as pd
import numpy as np

# Sample data for students
data = {
    'ID': range(1, 31),
    'Student_Age': [20, 21, 22, 19, 20, 18, 21, 20, 19, 22, 18, 19, 20, 21, 19, 18, 20, 21, 22, 19, 
                    20, 21, 18, 19, 20, 22, 18, 19, 20, 21],
    'Sex': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female',
            'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female',
            'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'High_School_Type': ['Private', 'State', 'Private', 'State', 'Private', 'State', 'Private', 'State', 'Private', 'State',
                         'Private', 'State', 'Private', 'State', 'Private', 'State', 'Private', 'State', 'Private', 'State',
                         'Private', 'State', 'Private', 'State', 'Private', 'State', 'Private', 'State', 'Private', 'State'],
    'Scholarship': ['50%', '50%', '75%', '50%', '50%', '50%', '50%', '50%', '75%', '50%', '50%', '50%', '75%', '50%', '50%', 
                    '50%', '75%', '50%', '50%', '50%', '50%', '75%', '50%', '50%', '50%', '50%', '50%', '75%', '50%', '50%'],
    'Additional_Work': ['Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 
                        'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes'],
    'Sports_activity': ['Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 
                        'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes'],
    'Transportation': ['Private', 'Public', 'Private', 'Public', 'Private', 'Public', 'Private', 'Public', 'Private', 'Public',
                       'Private', 'Public', 'Private', 'Public', 'Private', 'Public', 'Private', 'Public', 'Private', 'Public',
                       'Private', 'Public', 'Private', 'Public', 'Private', 'Public', 'Private', 'Public', 'Private', 'Public'],
    'Weekly_Study_Hours': [10, 8, 12, 9, 11, 7, 13, 10, 9, 11, 6, 10, 8, 12, 9, 11, 7, 13, 10, 9, 
                            11, 6, 10, 8, 12, 9, 11, 7, 13, 10],
    'Attendance': ['Always', 'Sometimes', 'Always', 'Sometimes', 'Always', 'Sometimes', 'Always', 'Sometimes', 'Always', 'Sometimes',
                   'Always', 'Sometimes', 'Always', 'Sometimes', 'Always', 'Sometimes', 'Always', 'Sometimes', 'Always', 'Sometimes',
                   'Always', 'Sometimes', 'Always', 'Sometimes', 'Always', 'Sometimes', 'Always', 'Sometimes', 'Always', 'Sometimes'],
    'Reading': ['Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 
                'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes'],
    'Notes': ['Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 
              'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes'],
    'Listening_in_Class': ['No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 
                           'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No'],
    'Project_work': ['No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 
                      'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No'],
    'Grade': ['AA', 'BB', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'JJ', 'KK', 'LL', 'MM', 'NN', 'OO', 'PP', 
              'QQ', 'RR', 'SS', 'TT', 'UU', 'VV', 'WW', 'XX', 'YY', 'ZZ', 'AAA', 'BBB', 'CCC', 'DDD']
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to Excel
df.to_excel('students_data.xlsx', index=False)





# Generate sample data for business records
import pandas as pd
import numpy as np

# Generate sample data for business records
num_records = 50
data = {
    'Company_ID': range(1, num_records + 1),
    'Company_Name': [f'Company {i}' for i in range(1, num_records + 1)],
    'Industry': np.random.choice(['Technology', 'Retail', 'Finance', 'Healthcare', 'Manufacturing'], num_records),
    'Revenue': np.random.randint(100000, 10000000, size=num_records),
    'Employees': np.random.randint(10, 500, size=num_records),
    'Location': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], num_records)
}

# Create DataFrame
df_business = pd.DataFrame(data)

# Save to Excel
df_business.to_excel('business_data.xlsx', index=False)













#@@@@@@@@@@@@JSONS ############

import json
import numpy as np

# Generate sample data for business records
num_records = 50
data = {
    'business_records': [
        {
            'Company_ID': i + 1,
            'Company_Name': f'Company {i + 1}',
            'Industry': np.random.choice(['Technology', 'Retail', 'Finance', 'Healthcare', 'Manufacturing']),
            'Revenue': int(np.random.randint(100000, 10000000)),
            'Employees': int(np.random.randint(10, 500)),
            'Location': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'])
        }
        for i in range(num_records)
    ]
}

# Save to JSON file
with open('BSdata.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)



import json
import numpy as np

# Generate sample data for sports records
num_records = 50
data = {
    'sports_records': [
        {
            'Player_ID': i + 1,
            'Player_Name': f'Player {i + 1}',
            'Sport': np.random.choice(['Football', 'Basketball', 'Tennis', 'Golf', 'Soccer']),
            'Team': np.random.choice(['Team A', 'Team B', 'Team C', 'Team D', 'Team E']),
            'Age': int(np.random.randint(18, 40)),
            'Country': np.random.choice(['USA', 'Spain', 'Brazil', 'France', 'Germany'])
        }
        for i in range(num_records)
    ]
}

# Save to JSON file
with open('sports_data.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)
