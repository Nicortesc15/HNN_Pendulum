# HNN-Double-Pendulum

**Learning Hamiltonian dynamics from data with a double pendulum system.**

This project aims to model the Hamiltonian system of a double pendulum using machine learning techniques.  
We combine classical physics, numerical solvers, and neural networks to study how well models can learn and preserve physical properties such as energy conservation.

---

## Project Structure

- **`double_pendulum.py`**  
  Implements the physics and equations of motion for the double pendulum.

- **Numerical Solvers:**
  - `explicit_euler.py`: Explicit Euler solver for simulating the system.
  - `symplectic_euler.py`: Symplectic Euler solver that better conserves energy over long simulations.

- **Neural Networks:**
  - `FFNN.py` and `FFNN_utils.py`: A standard feed-forward neural network (FFNN) that learns the system's gradients directly from data.
  - `HNN.py` and `HNN_utils.py`: A Hamiltonian Neural Network (HNN) that predicts the Hamiltonian function, enforcing energy conservation.

- **Supporting Files:**
  - `constants.py`: Stores physical constants for the double pendulum system.
  - `utils.py`: Includes plotting and analysis functions.
  - `main.py`: Entry point to run simulations, train models, and visualize results.

---

## How It Works

1. **Physics-Based Simulation:**
   - Use the known double pendulum dynamics to generate trajectory data.
   - Analyze how numerical solvers behave (stability, energy drift).

2. **Learning-Based Approaches:**
   - **FFNN** learns gradients (forces) directly from the generated data.
   - **HNN** learns the Hamiltonian (total energy), ensuring better conservation laws.

3. **Comparison and Analysis:**
   - Evaluate how FFNN and HNN perform over time.
   - Compare energy conservation, stability, and prediction accuracy.
   - Visualize trajectories, energy evolution, and error metrics.