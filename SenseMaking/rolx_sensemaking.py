#  This file will implement KDD 2012, sense making (NodeSense specifically), as for our current data Neighbors will provide little insight.
#  This is to clarify the semantics of the 4 roles given by RolX in my thesis experiment
#  Exerpt from Paper https://dl.acm.org/doi/10.1145/2339530.2339723
#
# NodeSense takes as input RolX ’s node-by-role matrix, G,
# and a matrix of node measurements, M. 
#
# -- my part
#  For HPC_Analytics Experiment the node measurements consisted of 
#  -- 
# 
# NodeSense then computes a nonnegative matrix E such that G·E ≈ M. The matrix E represents the role contribution to node measurements. A default
# matrix E
# ′
# is also computed by using G
# ′ = ones(n, 1), where
# the n nodes belong to one role. Then, for each role r and
# for each measurement s, NodeSense computes E(r,s)
# E′(r,s)
# . This
# ratio provides the role-contribution to node-measurements
# compared to the default contribution
#

import pandas as pd

# to be M
node_measurements = pd.read_csv('/Users/luis/Documents/Spring 2020/Current_Topics_AI/Code/Data/features_extracted.csv')
print(node_measurements.shape)

# to be G
node_by_role = pd.read_csv('/Users/luis/Documents/Spring 2020/Current_Topics_AI/Code/Data/transposed_roles.csv')
print(node_by_role.shape)


def make_sense(G, M):
    pass
