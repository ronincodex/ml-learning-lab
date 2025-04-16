import plotly.graph_objects as go

# Define the Chomsky hierarchy levels, RNN types, and curriculum steps
chomsky_levels = ["Regular", "Context-Free", "Context-Sensitive", "Recursively Enumerable"]
rnn_types = ["Simple RNN", "GRU", "Transformer"]
curriculum_steps = ["Simple Repititionos", "Nested Patterns", "Recursive Patterns"]

# Map each node to a position in 3D space
x_vals = list(range(len(chomsky_levels))) # X-axis for Chomsky Hierarchy
y_vals = list(range(len(rnn_types))) # Y-axis for RNN types
z_vals = list(range(len(curriculum_steps))) # Z-axis for curriculum learning steps

# Generate node positoins
nodes = []
edges = []
for i, chomsky in enumerate(chomsky_levels):
    for j, rnn in enumerate(rnn_types):
        for k, curriculum in enumerate(curriculum_steps):
            nodes.append((i, j, k, chomsky, rnn, curriculum))
            # Create edges to visualize relationships
            if j > 0: # Link to previous RNN type for hierarchy
                edges.append(((i, j - 1, k), (i, j, k)))
            if k > 0: # Link to previoius curriculum step for sequence progression
                edges.append(((i, j, k-1), (i, j, k)))
            if i > 0: # Link to previous Chomsky level for language complexity progression
                edges.append(((i -1, j, k), (i, j, k)))

# Separate the node coordination for plotting
x_nodes = [node[0] for node in nodes]
y_nodes = [node[1] for node in nodes]
z_nodes = [node[2] for node in nodes]
node_labels = [f"{node[3]}, {node[4]}, {node[5]}" for node in nodes]

# Create edges
x_edges = []
y_edges = []
z_edges = []
for edge in edges:
    x_edges += [edge[0][0], edge[1][0], None] # X-coordinates
    y_edges += [edge[0][1], edge[1][1], None] # Y-coordinates
    z_edges += [edge[0][2], edge[1][2], None] # Z-coordinates

# Initialize figure
fig = go.Figure()

# Add nodes
fig.add_trace(go.Scatter3d(x=x_nodes, y=y_nodes, z=z_nodes, mode='markers+text', text=node_labels, marker=dict(size=8, color='blue'), textposition="top center"))

# Add edges 
fig.add_trace(go.Scatter3d(x=x_edges, y=y_edges, z=z_edges, mode='lines', line=dict(color='gray', width=2)))

# Set axis labels and titles
fig.update_layout(
    title="RNNs, Chomsky Hierarchy, and Curriculum Learning",
    scene=dict(
        xaxis_title="Chomsky Hierarchy (Language Complexity)",
        yaxis_title="RNN Type (Model Capability)",
        zaxis_title="Curriculum Learning Steps",
        xaxis=dict(tickvals=list(range(len(chomsky_levels))), ticktext=chomsky_levels),
        yaxis=dict(tickvals=list(range(len(rnn_types))), ticktext=rnn_types),
        zaxis=dict(tickvals=list(range(len(curriculum_steps))), ticktext=curriculum_steps)
    )
)

# Display the plot
fig.show()
