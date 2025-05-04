import torch
import constants

# Constants for the single pendulum

M = constants.M  # Mass pendulum
L = constants.L  # Length pendulum
G = constants.G  # Gravitational constant

# Hamiltonian function
def hamiltonian(states: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hamiltonian (total energy) of the single pendulum.

    Args:
        states (torch.Tensor): Tensor of shape (N, 2), where N is the number of samples.
                               Each row contains [q, p] (angle and momentum).

    Returns:
        torch.Tensor: Tensor of shape (N,), where each entry is the Hamiltonian (total energy) of a state.
    """
    # Extract states
    states = states.detach()
    if states.ndim == 1:
        q, p = states[0], states[1]
    else:
        q, p = states[:, 0], states[:, 1]

    # Kinetic energy
    e_kin = p**2 / (2 * M * L**2)

    # Potential energy
    e_pot = M * G * L * (1 - torch.cos(q))

    # Total Hamiltonian
    h = e_kin + e_pot

    return h

# Vector field function
def vector_field(states: torch.Tensor) -> torch.Tensor:
    """
    Compute the vector field of the single pendulum.

    Args:
        states (torch.Tensor): Tensor of shape (N, 2), where N is the number of samples.
                               Each row contains [q, p] (angle and momentum).

    Returns:
        torch.Tensor: Tensor of shape (N, 2), where each entry represents the time derivatives.
    """

    # Extract states
    states = states.detach()
    if states.ndim == 1:
        q, p = states[0], states[1]
    else:
        q, p = states[:, 0], states[:, 1]

    # Compute time derivatives from Hamilton's equations
    dq_dt = p / M
    dp_dt = -M * G * L * torch.sin(q)

    # Combine derivatives into a single tensor
    derivatives = torch.stack([dq_dt, dp_dt], dim=-1)

    return derivatives

# Monte Carlo sampling for the single pendulum
def monte_carlo_sampling(q_range = (-torch.pi, torch.pi), p_range = (-1, 1), samples = 1000) -> dict:
    """
    Generate training data for a NN using Monte Carlo sampling.

    Args:
        q_range (tuple): Range of q values.
        p_range (tuple): Range of p values.
        samples (int): Number of samples to generate.

    Returns:
        dict: Dictionary containing training data for NN.
    """

    # Randomly sample states
    q_samples = torch.empty(samples).uniform_(*q_range)
    p_samples = torch.empty(samples).uniform_(*p_range)

    # Combine sampled states into a tensor
    states = torch.stack([q_samples, p_samples], dim=1)

    # Compute derivatives for all sampled states
    derivatives = vector_field(states)

    # Return data as a dictionary
    return {'states': states, 'derivatives': derivatives}