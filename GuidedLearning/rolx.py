#  RolX used in MS Thesis.

import networkx
from GuidedLearning import neo4j as ts
import pandas as pd
import warnings
from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

from graphrole import RecursiveFeatureExtractor, RoleExtractor

def rs2graph(rs):
    graph = networkx.MultiDiGraph()

    for record in rs:
        node = record['n']
        if node:
            print("adding node")
            nx_properties = {}
            nx_properties.update(node._properties)
            nx_properties['labels'] = node.labels
            graph.add_node(node.id, **nx_properties)
        relationship = record['r']
        if relationship is not None:   # essential because relationships use hash val
            print("adding edge")
            graph.add_edge(
                relationship.start, relationship.end, key=relationship.type
            )

    return graph


uri = "bolt://localhost:7687"

user = "neo4j"
password = "12345"

session = ts.HPCJobDatabase(uri, user, password)
rs = session.query_small_set()

G = rs2graph(rs)

session.close()

feature_extractor = RecursiveFeatureExtractor(G)
features = feature_extractor.extract_features()

print(f'\nFeatures extracted from {feature_extractor.generation_count} recursive generations:')
print(features)
# assign node roles
role_extractor = RoleExtractor(n_roles=None)
role_extractor.extract_role_factors(features)
node_roles = role_extractor.roles

print('\nNode role assignments:')
pprint(node_roles)

print('\nNode role membership by percentage:')
print(role_extractor.role_percentage.round(2))

# PLOTTING

# build color palette for plotting
unique_roles = sorted(set(node_roles.values()))
color_map = sns.color_palette('Paired', n_colors=len(unique_roles))
# map roles to colors
role_colors = {role: color_map[i] for i, role in enumerate(unique_roles)}
# build list of colors for all nodes in G
node_colors = [role_colors[node_roles[node]] for node in G.nodes]

# plot graph
plt.figure()

with warnings.catch_warnings():
    # catch matplotlib deprecation warning
    warnings.simplefilter('ignore')
    nx.draw(
        G,
        pos=nx.spring_layout(G, seed=42),
        with_labels=True,
        node_color=node_colors,
    )

plt.show()