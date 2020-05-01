#  This file will implement KDD 2012, sense making (NodeSense specifically), as for our current data Neighbors will provide little insight.
#  This is to clarify the semantics of the 4 roles given by RolX in my thesis experiment

import pandas as pd
import numpy as np

#  UNUSED FILE (CURRENTLY)
#  /Users/luis/Documents/Spring 2020/Current_Topics_AI/Code/Data/node_role_assignment.csv

# to be N (new attempt)
node_by_role_percentages = pd.read_csv('/Users/luis/Documents/Spring 2020/Current_Topics_AI/Code/Data/node_role_membership_by_percentage.csv')

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
n_array = node_by_role_percentages.to_numpy()

'''
NodeSense takes as input RolX ’s node-by-role matrix, G, and a matrix of node measurements, M. In this case they are 
total_degree, in_degree, internal_edges, out_degree, external_edges
NodeSense then computes a nonnegative matrix E such that G·E ≈ M. The matrix E represents the role contribution to node measurements. 
A default matrix E′ is also computed by using G′ = ones(n, 1), where the n nodes belong to one role. 
Then, for each role r and for each measurement s, NodeSense computes E(r,s) / E′(r,s).
This ratio provides the role-contribution to node-measurements compared to the default contribution.
Exerpt from Paper https://dl.acm.org/doi/10.1145/2339530.2339723
'''
def Node_Sense(G, M, X):
    E = np.dot(M.T, G)
    E_non_roles = E[:,1]
    # g_prime = np.ones((60307,1))
    g_prime = X
    E_prime = np.dot(M.T, g_prime)
    ratios = []
    # e_first_row = E[:,1]
    e_prime_first_row = E_prime[:,0]
    for i in range(1,6):
        for j in range(1,2):
            temp = []
            temp.append(E[i][j] / E_prime[i][j])
        ratios.append(temp)

    return(E, ratios)


# results = Node_Sense(g_array, m_array)
results = Node_Sense(n_array, m_array, g_array)
e_dataframe = pd.DataFrame(results[0])
ratios_dataframe = pd.DataFrame(results[1])
e_dataframe.to_csv('/Users/luis/Documents/Spring 2020/Current_Topics_AI/Code/Results/role_contribution_to_node_measurements.csv')
ratios_dataframe.to_csv('/Users/luis/Documents/Spring 2020/Current_Topics_AI/Code/Results/role-contribution_to_node-measurements_compared_to_default.csv')