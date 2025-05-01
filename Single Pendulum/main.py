import torch
import torch.nn as nn

import utils
import constants
import single_pendulum

import FFNN.FFNN as FFNN
import FFNN.FFNN_utils as FFNN_utils
import HNN.HNN as HNN
import HNN.HNN_utils as HNN_utils

import solvers.explicit_euler as explicit_euler
import solvers.symplectic_euler as symplectic_euler
from single_pendulum import hamiltonian

# Convert initial state into a torch tensor
Y0 = torch.tensor(constants.Y0, dtype=torch.float32)
T_span = constants.T_span

def solve_numerically(selected_solver: str) -> None:
    """
    Solve the system with a selected solver.

    Args:
        selected_solver (str): The name of the solver to use. Options: "Explicit Euler", "Symplectic Euler"

    Returns:
        None
    """

    # Set the numerical solver stated in the select_solver argument
    match selected_solver:
        case "Explicit Euler":
            func = explicit_euler.solve
        case "Symplectic Euler":
            func = symplectic_euler.solve
        case _:
            raise ValueError(f"{selected_solver} is not a valid solver.")

    # Solve the system for initial state Y0 and time span t_span
    t_values, y_values = func(single_pendulum.vector_field, "_", Y0, T_span)

    # Plot the complete trajectory for the double pendulum over the whole time range
    utils.plot_positions_in_cartesian(t_values, y_values, selected_solver)

    # Plot the Hamiltonian over time to see if it stays constant
    utils.plot_hamiltonian_deviation_over_time(t_values, y_values, selected_solver)

def learn_hamiltonian_and_solve(selected_model: str) -> nn.Module:
    """
    Learn the Hamiltonian function from data and use the learned model to solve the system with the selected_model argument.

    Args:
          selected_model (str): Model which should be used for learning the Hamiltonian

    Returns:
        nn.Module: Trained model.
    """

    data_samples = single_pendulum.monte_carlo_sampling(samples=1000)
    X_train, Y_train = data_samples['states'], data_samples['derivatives']

    print(f"\n --- Start Training of {selected_model} --- \n")

    match selected_model:
        case "FFNN":
            model = FFNN.FFNN(input_dim=2, hidden_dim=128, output_dim=2)
            loss_history = FFNN_utils.train_ffnn(
                model=model,
                num_epochs=300,
                X=X_train,
                Y=Y_train
            )
        case "HNN":
            # Simulate "measuring" of data points for the supervised (data) loss
            X_measured = single_pendulum.monte_carlo_sampling(samples = 10) ['states']
            H_measured = hamiltonian(X_measured)

            model = HNN.HNN(input_dim=2, hidden_dim=64, output_dim=1)
            loss_history = HNN_utils.train_hnn(
                model=model,
                num_epochs=500,
                X_train=X_train,
                Y_train=Y_train,
                X_measured=X_measured,
                H_measured=H_measured
            )
        case _:
            raise ValueError(f"{selected_model} is not a valid model.")

    # Plot loss function over training epochs
    utils.plot_losses(loss_history, selected_model)

    # Use the trained network and solve with Symplectic Euler solver
    model.eval()
    t_values, y_values = symplectic_euler.solve(model, selected_model, Y0, T_span)

    # Plot the complete trajectory for the single pendulum over the whole time range
    utils.plot_positions_in_cartesian(t_values, y_values, f"{selected_model} with Symplectic Euler")

    # Plot the Hamiltonian over time to see if it stays constant
    utils.plot_hamiltonian_deviation_over_time(t_values, y_values, f"{selected_model} with Symplectic Euler")

    return model

if __name__ == "__main__":
    # Set the numerical solver for solving the known system
    use_solver = "Explicit Euler" # Alternatives: "Symplectic Euler" or "Explicit Euler"

    # Numerically solve the known system with the selected solver
    solve_numerically(use_solver)

    # Set the model to learn the Hamiltonian
    use_model = "HNN" # Alternatives: "HNN", "FFNN"

    # Learn the Hamiltonian and solve the learned system with the Symplectic Euler method
    trained_model = learn_hamiltonian_and_solve(use_model)

    if use_model == "HNN":
        # Plot the true and the learned Hamiltonian
        utils.compare_hamiltonian_single_pendulum(trained_model)
