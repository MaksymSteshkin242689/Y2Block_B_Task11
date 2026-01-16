# Task 11: Reinforcement Learning Controller for OT-2 Robot

**Author**: Maksym Steshkin  
**Student ID**: 242689  
**Date**: January 2026

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Implementation Steps](#implementation-steps)
3. [Design Choices](#design-choices)
4. [Tuning Strategies](#tuning-strategies)
5. [Performance Metrics](#performance-metrics)
6. [Comparison to PID Controller](#comparison-to-pid-controller)
7. [Best Hyperparameters](#best-hyperparameters)
8. [Model Weights](#model-weights)
9. [Files Description](#files-description)
10. [Usage Instructions](#usage-instructions)

---

## 🎯 Overview

This task implements a Reinforcement Learning (RL) controller using **Stable Baselines 3** to control the Opentrons OT-2 robot. The RL agent learns to move the pipette tip to any position within the work envelope through trial and error, optimizing its policy based on a reward function.

### Key Achievements

- ✅ **Gymnasium Environment**: Custom wrapper for OT-2 simulation
- ✅ **PPO Algorithm**: Proximal Policy Optimization for stable training
- ✅ **Sub-5mm Accuracy**: Achieves 3.2mm average positioning error
- ✅ **Fast Execution**: 76% fewer steps than PID controller
- ✅ **Comprehensive Testing**: Evaluation across multiple metrics

### Client Requirements

| **Positioning Error** | **Points** | **Achieved** |
|-----------------------|------------|--------------|
| < 1mm                 | 8 points   | ❌ 0% (3.2mm avg) |
| 1-5mm                 | 6 points   | ✅ **100%** |
| 5-10mm                | 4 points   | ✅ 100% |
| ≥ 10mm                | 0 points   | ✅ 0% |

**Current Score**: **6/8 points** (all targets within 5mm)

---

## 🏗️ Implementation Steps

### Step 1: Gymnasium Environment Wrapper

**File**: `maksym_steshkin_ot2_gym_wrapper.py`

Created a custom Gymnasium environment that wraps the OT-2 simulation:

```python
class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=300, target_threshold=0.005):
        # Initialize simulation
        self.sim = Simulation(num_agents=1, render=render)
        
        # Define spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,))
        
        # Workspace bounds
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1700])
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897])
```

**Key Components**:

1. **Action Space** (3D continuous):
   - `[vel_x, vel_y, vel_z]` ∈ [-1, 1]
   - Scaled to actual velocities: ±2.0 m/s

2. **Observation Space** (6D normalized):
   - `[current_x, current_y, current_z, goal_x, goal_y, goal_z]`
   - Normalized to [-1, 1] range

3. **Reward Function**:
   ```python
   reward = -0.1 (time penalty)
          + -10.0 * distance (distance penalty)
          + 50.0 (success bonus if distance < threshold)
   ```

4. **Termination Conditions**:
   - Success: Distance to goal < 5mm
   - Timeout: Max steps (300) reached

### Step 2: Wrapper Validation

**File**: `test_wrapper.py`

Comprehensive testing script that validates:
- ✅ Gymnasium interface compliance
- ✅ Action/observation space correctness
- ✅ Episode termination logic
- ✅ Random action baseline performance

**Test Results**:
- Random actions: ~0% success rate
- Proper reset/step format
- Observation bounds validated (with warning for minor violations)

### Step 3: RL Agent Training

**File**: `maksym_steshkin_train_ot2.py`

Trained PPO agent using **ClearML** for experiment tracking:

```python
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=3e-4,
    batch_size=64,
    n_steps=2048,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1
)

model.learn(total_timesteps=500000, callback=ot2_callback)
```

**Training Process**:
- Platform: GPU server with ClearML queue
- Duration: ~6 hours
- Total timesteps: 500,000
- Framework: Stable Baselines 3 + TensorFlow

### Step 4: Model Evaluation

**File**: `test_rl.py`

Comprehensive evaluation system that measures:
- Positioning accuracy (error in mm)
- Success rates at multiple thresholds
- Episode completion time
- Trajectory visualization

### Step 5: Demonstration & Visualization

**File**: `demonstrate_rl_agent.py`

Interactive demonstration with PyBullet rendering for:
- GIF creation
- Visual debugging
- Performance validation

---

## 🎨 Design Choices

### 1. Observation Space: Relative vs Absolute

**Choice**: Used **normalized absolute positions** (current + goal)

**Rationale**:
- ✅ Simple and effective
- ✅ Symmetric observation space
- ✅ No complex relative coordinate calculations
- ✅ Works well with PPO's policy network

**Alternative considered**: Relative position (goal - current)
- ❌ Would lose absolute position information
- ❌ May make learning harder

### 2. Action Space: Position vs Velocity

**Choice**: **Velocity control** (continuous actions)

**Rationale**:
- ✅ Smooth, continuous motion
- ✅ Natural for PPO (continuous action space)
- ✅ Allows for trajectory optimization
- ✅ Matches simulation's velocity-based control

**Alternative considered**: Direct position commands
- ❌ Would be discrete or require interpolation
- ❌ Less natural for learning smooth trajectories

### 3. Reward Function: Dense vs Sparse

**Choice**: **Dense reward** (distance-based + time penalty)

**Rationale**:
- ✅ Provides continuous feedback
- ✅ Guides agent toward goal
- ✅ Faster learning than sparse rewards
- ✅ Encourages both accuracy and speed

**Components**:
- Time penalty (-0.1): Discourages slow movement
- Distance penalty (-10 × dist): Minimizes positioning error
- Success bonus (+50): Strong incentive for goal achievement

### 4. Algorithm: PPO vs SAC vs TD3

**Choice**: **PPO** (Proximal Policy Optimization)

**Rationale**:
- ✅ Stable and reliable
- ✅ Works well with continuous actions
- ✅ Good sample efficiency
- ✅ Easy to tune

**Alternatives considered**:
- SAC: Better for precision but more complex
- TD3: Similar to SAC, focuses on continuous control
- May explore these in future iterations

### 5. Target Threshold: 1mm vs 5mm

**Choice**: **5mm threshold** for training

**Rationale**:
- ✅ Easier initial learning target
- ✅ Faster convergence
- ⚠️ Trade-off: Lower final precision

**Impact**: Agent achieves 100% success at 5mm but 0% at 1mm
**Future work**: Retrain with 1mm threshold or curriculum learning

---

## 🔧 Tuning Strategies

### Hyperparameter Search

**Approach**: Grid search across key hyperparameters

**Search Space**:
- Learning rate: [1e-4, 3e-4, 1e-3]
- Batch size: [32, 64, 128]
- N-steps: [1024, 2048, 4096]
- Target threshold: [1mm, 5mm, 10mm]

### Final Hyperparameters

Selected based on training stability and convergence:

| **Parameter** | **Value** | **Rationale** |
|---------------|-----------|---------------|
| Learning Rate | **3e-4** | Standard PPO lr, stable convergence |
| Batch Size | **64** | Good balance: speed vs stability |
| N-Steps | **2048** | Standard PPO buffer size |
| Gamma (discount) | **0.99** | Long-term planning important |
| GAE Lambda | **0.95** | Variance reduction |
| Clip Range | **0.2** | Standard PPO clipping |
| N-Epochs | **10** | Multiple updates per batch |
| Max Steps | **300** | Episode timeout |
| Target Threshold | **5mm** | Training convergence target |

### Tuning Process

1. **Baseline Run** (lr=3e-4, default params)
   - Result: Converged but 5mm accuracy

2. **Learning Rate Sweep**
   - 1e-4: Too slow
   - 3e-4: ✅ Good balance
   - 1e-3: Unstable

3. **Batch Size Adjustment**
   - 32: Noisy updates
   - 64: ✅ Stable
   - 128: Slower, minimal improvement

4. **Threshold Experiment**
   - 10mm: Too easy, agent exploits
   - 5mm: ✅ Good learning signal
   - 1mm: Doesn't converge (needs more training)

### Libraries Used

- **Stable Baselines 3** (v2.x): RL algorithms
- **Gymnasium** (v0.29+): Environment standard
- **PyBullet**: Physics simulation
- **NumPy**: Numerical operations
- **TensorFlow**: Neural network backend
- **ClearML**: Experiment tracking
- **Matplotlib**: Visualization

---

## 📊 Performance Metrics

### Evaluation Results (100 episodes)

**File**: `test_rl.py` generates comprehensive metrics

#### Accuracy Metrics

| **Metric** | **Value** |
|------------|-----------|
| **Mean Error** | **3.155 mm** |
| Std Dev Error | 1.013 mm |
| Min Error | 1.678 mm |
| Max Error | 4.261 mm |
| Median Error | 3.100 mm |

#### Success Rates

| **Threshold** | **Success Rate** | **Client Points** |
|---------------|------------------|-------------------|
| < 1mm | **0%** | 0/8 points |
| < 5mm | **100%** ✅ | **6/8 points** |
| < 10mm | 100% | N/A |

#### Efficiency Metrics

| **Metric** | **Value** |
|------------|-----------|
| Mean Episode Length | 36.8 steps |
| Mean Execution Time | 0.008 s |
| Avg Success Time | N/A (0 successes at 1mm) |

#### Client Score

**Final Score**: **6/8 points** (all targets within 5mm)

---

## ⚖️ Comparison to PID Controller

### Head-to-Head Benchmark

**Test**: 4 root tips from test_image_01.png

| **Metric** | **RL Controller** | **PID Controller** | **Winner** |
|------------|-------------------|-------------------|------------|
| **Mean Error** | 3.155 mm | **0.900 mm** ✅ | **PID** |
| **Success Rate (< 1mm)** | 0% | **100%** ✅ | **PID** |
| **Mean Steps** | **36.8** ✅ | 150.3 | **RL** |
| **Mean Time** | **0.008s** ✅ | 0.013s | **RL** |
| **Consistency** | ±1.013mm | **±0.0003mm** ✅ | **PID** |
| **Client Score** | 6/8 | **7/7** ✅ | **PID** |

### Analysis

#### RL Advantages ✅
- **76% fewer steps**: Much more efficient trajectories
- **38% faster execution**: Quicker per-inoculation time
- **Learned behavior**: Adapts through training

#### RL Disadvantages ❌
- **Lower accuracy**: 3.2mm vs 0.9mm (3.5× worse)
- **Higher variance**: Less consistent positioning
- **0% success at 1mm**: Doesn't meet highest client requirement
- **Training required**: 6 hours on GPU server

#### PID Advantages ✅
- **Excellent accuracy**: 0.9mm average error
- **Perfect consistency**: ±0.0003mm std dev
- **100% success at 1mm**: Meets highest client requirement
- **Deterministic**: Predictable behavior
- **No training**: Works immediately with tuned gains

#### PID Disadvantages ❌
- **More steps**: 150 vs 37 (4× more)
- **Slower convergence**: Takes longer to settle
- **Fixed parameters**: No adaptation

### Recommendation

**For Production**: **Use PID Controller** ✅

**Rationale**:
- PID meets client requirements (< 1mm)
- RL requires retraining for higher precision
- Accuracy more important than speed for this application

**Future Work**: 
- Retrain RL with 1mm threshold
- Explore hybrid approach: RL for coarse, PID for fine

---

## 🏆 Best Hyperparameters

### Final Configuration

**Model**: `260114.2210_maksym_steshkin_lr3e-4_b64_s2048_th5mm.zip`

```python
# Training Hyperparameters
LEARNING_RATE = 3e-4
BATCH_SIZE = 64
N_STEPS = 2048
TOTAL_TIMESTEPS = 500000
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
N_EPOCHS = 10

# Environment Configuration
MAX_STEPS_TRUNCATE = 300
TARGET_THRESHOLD = 0.005  # 5mm
MAX_VELOCITY = 2.0  # m/s

# Reward Function Coefficients
TIME_PENALTY = -0.1
DISTANCE_PENALTY_COEFF = -10.0
SUCCESS_BONUS = 50.0
```

### Training Details

- **Total Training Time**: ~6 hours
- **Platform**: GPU server (ClearML queue)
- **Docker Image**: `deanis/2023y2b-rl:latest`
- **Framework**: Stable Baselines 3 v2.x
- **Convergence**: ~300k timesteps
- **Final Success Rate**: 100% (at 5mm threshold)

### Performance vs Baseline

| **Configuration** | **Mean Error** | **Success @ 5mm** |
|-------------------|----------------|-------------------|
| Random Actions | ~50mm | 0% |
| **Trained PPO** | **3.2mm** | **100%** |
| Target (1mm) | 1.0mm | TBD (needs retraining) |

---

## 💾 Model Weights

### Location

**Best Individual Model**:
```
Task 11/models/260114.2210_maksym_steshkin_lr3e-4_b64_s2048_th5mm.zip
```

**Filename Format**: `YYMMDD.HHMM_name_lr{lr}_b{batch}_s{steps}_th{threshold}mm.zip`

### Model Information

- **Size**: ~2.5 MB
- **Architecture**: MLP Policy (2 hidden layers, 64 units each)
- **Input**: 6D observation (normalized positions)
- **Output**: 3D action (velocity commands)
- **Framework**: Stable Baselines 3 (PPO)

### Loading Model

```python
from stable_baselines3 import PPO

# Load trained model
model = PPO.load("models/260114.2210_maksym_steshkin_lr3e-4_b64_s2048_th5mm.zip")

# Use for prediction
action, _states = model.predict(observation, deterministic=True)
```

### Group Model Weights

**Note**: Individual model performed well, no group model submitted

---

## 📁 Files Description

### Core Implementation Files

1. **`maksym_steshkin_ot2_gym_wrapper.py`** (200 lines)
   - Custom Gymnasium environment
   - Action/observation spaces
   - Reward function
   - Termination conditions
   - Workspace bounds validation

2. **`test_wrapper.py`** (281 lines)
   - Environment validation
   - Random action baseline
   - Gymnasium compliance checks
   - Statistics reporting

3. **`maksym_steshkin_train_ot2.py`** (206 lines)
   - PPO training script
   - ClearML integration
   - Custom callback for metrics
   - Model saving and uploading

4. **`test_rl.py`** (481 lines)
   - Comprehensive evaluation
   - Performance metrics calculation
   - Visualization generation
   - Client scoring

5. **`demonstrate_rl_agent.py`** (355 lines)
   - Interactive demonstration
   - PyBullet rendering
   - GIF creation support
   - Multiple demonstration modes

### Supporting Files

- **`sim_class.py`**: OT-2 simulation wrapper
- **`ot_2_simulation_v6.urdf`**: Robot URDF file
- **`meshes/`**: Robot 3D models
- **`textures/`**: Petri dish textures

### Output Files

- **`models/260114.2210_maksym_steshkin_lr3e-4_b64_s2048_th5mm.zip`**: Trained model
- **`maksym_steshkin_evaluation_results.png`**: 6-plot evaluation visualization
- **`maksym_steshkin_trajectories.png`**: Sample trajectory plots
- **Screen recordings**: GIF demonstrations

---

## 🚀 Usage Instructions

### 1. Test Wrapper (Validation)

```bash
cd "Task 11"
python test_wrapper.py
```

**Output**: Validation checks + random action baseline

### 2. Train RL Agent

```bash
python maksym_steshkin_train_ot2.py \
    --learning_rate 0.0003 \
    --batch_size 64 \
    --n_steps 2048 \
    --total_timesteps 500000 \
    --max_steps_truncate 300 \
    --target_threshold 0.005
```

**Note**: Runs on ClearML queue (GPU server)

### 3. Evaluate Trained Model

```bash
python test_rl.py
```

**Output**:
- Console: Detailed metrics
- `maksym_steshkin_evaluation_results.png`: 6 comparison plots
- `maksym_steshkin_trajectories.png`: 4 sample trajectories

### 4. Demonstrate Agent (with Rendering)

```bash
python demonstrate_rl_agent.py
```

**Options**:
- Multiple episodes mode
- Single episode (for GIF)
- Custom target mode

**Use for**: Recording GIFs with screen capture

---

## 📈 Visualizations

### Generated Plots

1. **Error Distribution Histogram** (from `test_rl.py`)
   - Shows spread of positioning errors
   - Client requirement thresholds marked

2. **Success Rates Bar Chart**
   - < 1mm, < 5mm, < 10mm thresholds
   - Color-coded by achievement

3. **Episode Length Over Time**
   - Shows learning stability
   - Mean length indicated

4. **Positioning Error Over Episodes**
   - Error consistency check
   - Outlier detection

5. **Cumulative Error Distribution**
   - CDF plot
   - Percentage below each threshold

6. **Client Score Breakdown (Pie Chart)**
   - Distribution of score categories
   - Average score displayed

7. **Sample Trajectories (4 plots)**
   - Best performance
   - Median performance
   - Worst performance
   - Random sample

---

## 🔬 Future Improvements

### To Achieve < 1mm Accuracy

1. **Retrain with Tighter Threshold**:
   - Change `target_threshold` from 5mm to 1mm
   - Increase training timesteps: 500k → 2-5M
   - Add precision bonus to reward function

2. **Adjust Reward Function**:
   ```python
   # Current
   reward = -0.1 - 10*dist + 50*(dist<5mm)
   
   # Proposed
   reward = -0.1 - 100*dist + 100*(dist<1mm) + 50*(dist<5mm)
   ```

3. **Try Different Algorithms**:
   - **SAC** (Soft Actor-Critic): Better for precision
   - **TD3** (Twin Delayed DDPG): Continuous control specialist

4. **Curriculum Learning**:
   - Start with 10mm threshold
   - Gradually reduce to 5mm → 2mm → 1mm
   - Prevents training collapse

5. **Hybrid Approach**:
   - Use RL for fast approach (coarse positioning)
   - Switch to PID for final precision (< 5mm)
   - Combine RL speed with PID accuracy

---

## ✅ Requirements Checklist

### Deliverables

- ✅ `README.md` (this file) - comprehensive documentation
- ✅ `maksym_steshkin_ot2_gym_wrapper.py` - Gymnasium wrapper
- ✅ `test_wrapper.py` - wrapper testing script
- ✅ `maksym_steshkin_train_ot2.py` - training code
- ✅ `test_rl.py` - testing/evaluation code
- ✅ `demonstrate_rl_agent.py` - demonstration code
- ✅ Best model weights - saved in `models/`
- ✅ Visualizations - evaluation_results.png, trajectories.png
- ✅ GIF - screen recording available
- ⏳ Presentation slides - to be created

### Documentation Coverage

- ✅ Implementation steps and design choices
- ✅ Tuning strategies and libraries used
- ✅ Performance metrics with error analysis
- ✅ Comparison to PID controller
- ✅ Best hyperparameters documented
- ✅ Model weights location specified

---

## 📞 Contact & Support

**Author**: Maksym Steshkin  
**Student ID**: 242689  
**Course**: Data Science and AI  
**Institution**: Breda University of Applied Sciences

For questions about implementation or results, refer to code comments and this documentation.

---

**Status**: ✅ **COMPLETE** - All code implemented and tested  
**Achievement**: **6/8 points** (100% success at < 5mm)  
**Recommendation**: Retrain with 1mm threshold to achieve 8/8 points

**Last Updated**: January 15, 2026
