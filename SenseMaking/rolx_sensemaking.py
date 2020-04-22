#  This file will implement KDD 2012, sense making (NodeSense specifically), as for our current data Neighbors will provide little insight.
#  This is to clarify the semantics of the 4 roles given by RolX in my thesis experiment

import pandas as pd
import numpy as np

# to be M
node_measurements = pd.read_csv('/Users/luis/Documents/Spring 2020/Current_Topics_AI/Code/Data/features_extracted.csv')

# to be G
node_by_role = pd.read_csv('/Users/luis/Documents/Spring 2020/Current_Topics_AI/Code/Data/transposed_roles.csv')

# roles need to be cleaned

count = 0
new_col = {}
for x in node_by_role.role:
    new_col[count] = x
    count += 1

curr = node_by_role.drop(['role'], axis=1)
curr_data = pd.Series(new_col).to_frame('role')
newdf_ = pd.DataFrame(curr_data)
node_by_role_cleaned = pd.concat([curr, newdf_], axis=1)
node_by_role_cleaned.role[node_by_role_cleaned.role == 'role_0'] = 0
node_by_role_cleaned.role[node_by_role_cleaned.role == 'role_1'] = 1
node_by_role_cleaned.role[node_by_role_cleaned.role == 'role_2'] = 2
node_by_role_cleaned['role'] = node_by_role_cleaned['role'].astype(int)
m_array = node_measurements.to_numpy()
g_array = node_by_role_cleaned.to_numpy()

'''
NodeSense takes as input RolX ’s node-by-role matrix, G, and a matrix of node measurements, M. 
NodeSense then computes a nonnegative matrix E such that G·E ≈ M. The matrix E represents the role contribution to node measurements. 
A default matrix E′ is also computed by using G′ = ones(n, 1), where the n nodes belong to one role. 
Then, for each role r and for each measurement s, NodeSense computes E(r,s) / E′(r,s).
This ratio provides the role-contribution to node-measurements compared to the default contribution.
Exerpt from Paper https://dl.acm.org/doi/10.1145/2339530.2339723
'''
def Node_Sense(G, M):
    E = np.dot(M.T, G)
    g_prime = np.ones((60307,1))
    E_prime = np.dot(M.T, g_prime)
    ratios = []
    e_first_row = E[:,0]
    e_prime_first_row = E_prime[:,0]
    for i in range(6):
        temp = e_first_row[i] / e_prime_first_row[i]
        ratios.append(temp)

    return(E, ratios)


results = Node_Sense(g_array, m_array)
e_dataframe = pd.DataFrame(results[0])
ratios_dataframe = pd.DataFrame(results[1])
e_dataframe.to_csv('role_contribution_to_node_measurements.csv')
ratios_dataframe.to_csv('role-contribution_to_node-measurements_compared_to_default.csv')