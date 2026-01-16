## Project Structure

```
Project_Codes/
├── main.py                  # Main entry point for training/evaluation
├── run_experiments.py       # Curriculum learning experiment runner
├── analyze_results.py       # Script to analyze and visualize results
├── parameters/
│   ├── setup.json           # Main configuration file
│   ├── environments. json    # Environment configurations
│   └── hp.json              # Hyperparameter tuning configurations
├── fltlnd/
│   ├── agent.py             # RL agent implementations (DQN, PPO, Random)
│   ├── handler.py           # Main execution handler
│   ├── obs. py               # Observation wrappers (TreeObs)
│   ├── deadlocks.py         # Deadlock detection logic
│   ├── predict.py           # Path prediction utilities
│   ├── replay_buffer.py     # Experience replay buffers
│   ├── logger.py            # Logging utilities (WandB integration)
│   └── utils.py             # Utility functions
├── checkpoints/             # Saved model checkpoints
├── tmp/logs/                # Training logs and metrics
└── plots/                   # Generated plots from analysis
```

## Requirements

The following Python packages are required: 

* Python 3.8+
* PyTorch
* NumPy
* Flatland (`flatland-rl`)
* Pandas
* Matplotlib
* WandB (optional, for cloud logging)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/JiujiuLiao/Reinforcement-Learning-in-Railways-Rescheduling-Using-Flatland.git
   cd Reinforcement-Learning-in-Railways-Rescheduling-Using-Flatland/Project_Codes
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install torch numpy flatland-rl pandas matplotlib wandb
   ```
