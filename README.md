# control-toys
 
A collection of simple nonlinear dynamical systems with linear quadratic control

![control-toys](gifs/disturb_true/all.gif)

## Dependencies
- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Autograd](https://github.com/HIPS/autograd)

## Instructions
1. Tweak `settings.py`
2. Run `main.py`

## System definitions
- Systems are defined in `systems.py`
- System definitions consist of
  - `dynamics` function, a differentiable nonlinear map n+m → n that governs deterministic continuous-time state transitions
  - `cost` function, a differentiable map n+m → 1 that governs the control objective
  - `disturbance` function, a map n+m → n that generates stochastic additive disturbances
  - `x_eq_list` and `u_eq_list`, lists of equilibrium states and inputs to cycle thru
  - `make_artist_props` function, makes artist properties to draw the system in the current state-input configuration

## Functionality
- Dynamics are linearized and costs quadratized around the initial equilibrum point
- Steady-state LQR gain is designed on this basis
- Control inputs are generated to track the current equilibrium point
