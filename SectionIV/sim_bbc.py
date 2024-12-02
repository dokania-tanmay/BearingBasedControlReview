import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
num_agents = 5  # Number of agents
time_step = 2  # Time step for simulation
total_time = 200  # Total simulation time
num_steps = int(total_time / time_step)
random_radius = 1  # Target radius for the circle

# Control gain (to drive the agents to the circle)
k = 0.05

# Random initial positions for agents within a bounded space
np.random.seed(42)  # For reproducibility
initial_positions = np.random.rand(num_agents, 2) * 4 - 2  # Random positions in range [-2, 2]

# Initialize position history for visualization
position_history = np.zeros((num_steps, num_agents, 2))

# Function to compute the bearing (avoid division by zero)
def compute_bearing(rel_pos):
    norm = np.linalg.norm(rel_pos)
    if norm == 0:
        return np.array([0, 0])  # Return a zero vector if the agents are in the same position
    return rel_pos / norm

# Function to compute the control law (converge to a circle)
def control_law(positions):
    new_positions = np.copy(positions)
    # Compute center of mass (for simplicity, we aim for a circular formation around this center)
    center_of_mass = np.mean(positions, axis=0)
    
    # Control each agent to move towards the circle around the center of mass
    for i in range(num_agents):
        # Desired direction to the center
        rel_pos = positions[i] - center_of_mass
        bearing = compute_bearing(rel_pos)
        
        # Calculate desired position on the circle at the target radius
        desired_position = center_of_mass + bearing * random_radius
        
        # Move the agent towards its desired position
        new_positions[i] += time_step * k * (desired_position - positions[i])
    
    return new_positions

# Simulation loop to populate position history
positions = initial_positions
for t in range(num_steps):
    positions = control_law(positions)  # Apply control law
    position_history[t, :, :] = positions  # Save positions for each frame

# Create the plot
fig, ax = plt.subplots()
ax.set_xlim(-random_radius - 1, random_radius + 1)
ax.set_ylim(-random_radius - 1, random_radius + 1)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
# ax.set_title('Agents Converging to a Circle with Dotted Circle')

# Create scatter plots for each agent
scatters = [ax.scatter([], [], label=f'Agent {i+1}') for i in range(num_agents)]
lines = [ax.plot([], [], label=f'Agent {i+1}')[0] for i in range(num_agents)]

# List to keep track of the dotted circle (so we can remove the previous one)
dotted_circle = []

# Function to plot a fresh dotted circle at the current centroid
def plot_dotted_circle(ax, center, radius, num_points=100):
    # Generate points along the circle
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return ax.plot(x, y, 'k:', label='Desired Circle')  # 'k:' means black dotted line

# Initialize function for the animation
def init():
    for scatter in scatters:
        scatter.set_offsets([])  # Initialize with no data
    for line in lines:
        line.set_data([], [])  # Initialize with no data
    return scatters + lines

# Update function for each frame of the animation
def update(frame):
    # Remove the previous dotted circle if it exists
    global dotted_circle
    for line in dotted_circle:
        line.remove()  # Remove previous dotted circle
    
    # Update scatter plot (position of each agent)
    for i in range(num_agents):
        scatters[i].set_offsets(position_history[frame, i, :])

    # Update line plot (trajectories of each agent)
    for i in range(num_agents):
        lines[i].set_data(position_history[:frame, i, 0], position_history[:frame, i, 1])

    # Calculate the centroid (center of mass)
    centroid = np.mean(position_history[frame, :, :], axis=0)

    # Plot the fresh dotted circle around the centroid
    dotted_circle = plot_dotted_circle(ax, centroid, random_radius)

    return scatters + lines

# Create the animation
ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, interval=200, blit=False, repeat=False)

# Show the animation
plt.legend()

# Save the final frame as an image (screenshot of the last frame)
ani.event_source.stop()  # Stop the animation after it finishes
plt.savefig("final_frame.png")  # Save the screenshot

# Show the animation window
plt.show()
