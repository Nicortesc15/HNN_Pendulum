## HNN Single Pendulum

- **`single_pendulum.py`**  
  Contains the physical model of the single pendulum system (adapted from the double pendulum).

- **Numerical Solvers**  
  Used to numerically integrate the equations of motion and analyze properties such as energy conservation and long-term stability:
  - `explicit_euler.py`
  - `symplectic_euler.py`
  - `leapfrog.py`

- **Feedforward Neural Network (FFNN)**  
  Implements a standard feedforward neural network that directly learns the gradients from data samples:
  - `FFNN.py`
  - `FFNN_utils.py`  
  The learned gradients are then used with the numerical solvers to integrate the system's behavior.

- **Hamiltonian Neural Network (HNN)**  
  Learns to approximate the Hamiltonian function to better preserve physical properties like energy conservation:
  - `HNN.py`
  - `HNN_utils.py`  
  The learned Hamiltonian is then integrated using the same numerical solvers.

- **Other Components**
  - `constants.py`: Stores the physical constants of the pendulum system.
  - `main.py`: Entry point to run simulations and train models.
  - `utils.py`: Provides plotting and utility functions for analysis.
