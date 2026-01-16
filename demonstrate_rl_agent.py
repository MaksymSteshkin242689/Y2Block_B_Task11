"""
Demonstration Script for RL Agent Controlling OT-2 Robot

This script demonstrates the trained RL agent controlling the OT-2 robot to reach
target positions. It renders the simulation visually so you can see the robot
movement in real-time. This is useful for creating GIFs or videos showing the
agent's performance.

The script will:
1. Load the trained RL model
2. Run multiple episodes with different random target positions
3. Display the robot moving to each target in PyBullet visualization
4. Print statistics about positioning accuracy

Usage:
    python demonstrate_rl_agent.py

For creating GIFs:
    - Use screen recording software (QuickTime, OBS, etc.)
    - Or use the built-in save_gif option (requires imageio)

Requirements:
    - stable-baselines3
    - gymnasium
    - numpy
    - pybullet (for rendering)
    - sim_class.py (OT-2 simulation)
    - maksym_steshkin_ot2_gym_wrapper.py (custom environment)

Author: Maksym Steshkin
Date: January 2026
"""

import numpy as np
from stable_baselines3 import PPO
from maksym_steshkin_ot2_gym_wrapper import OT2Env
import time
from pathlib import Path


def demonstrate_agent(model_path, num_episodes=5, slow_motion=False, save_gif=False):
    """
    Demonstrate the trained RL agent controlling the robot.
    
    Parameters
    ----------
    model_path : str
        Path to the trained model file
    num_episodes : int
        Number of demonstration episodes (default: 5)
    slow_motion : bool
        If True, adds delays to slow down visualization (default: False)
    save_gif : bool
        If True, attempts to save frames as GIF (requires imageio) (default: False)
    """
    print("=" * 80)
    print("RL AGENT DEMONSTRATION - OT-2 ROBOT POSITIONING")
    print("=" * 80)
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"\n❌ Error: Model not found at {model_path}")
        print("Please ensure the trained model file exists.")
        return
    
    # Load the trained model
    print(f"\n[1/3] Loading trained model...")
    print(f"  Model: {model_path}")
    model = PPO.load(model_path)
    print("  ✓ Model loaded successfully")
    
    # Create environment with rendering enabled
    print(f"\n[2/3] Initializing environment with rendering...")
    env = OT2Env(render=True, max_steps=300, target_threshold=0.005)
    print("  ✓ Environment initialized")
    print("  ✓ PyBullet window opened")
    
    # Run demonstration episodes
    print(f"\n[3/3] Running {num_episodes} demonstration episodes...")
    print("-" * 80)
    
    all_errors = []
    all_steps = []
    
    for episode in range(num_episodes):
        print(f"\n📍 EPISODE {episode + 1}/{num_episodes}")
        
        # Reset environment and get initial observation
        obs, info = env.reset()
        
        # Get goal position for display
        goal = env.goal_position
        print(f"  Target position: X={goal[0]:.4f}m, Y={goal[1]:.4f}m, Z={goal[2]:.4f}m")
        
        # Episode tracking
        episode_done = False
        steps = 0
        trajectory = []
        
        # Run episode until completion
        while not episode_done:
            # Get action from trained model (deterministic for consistent behavior)
            action, _states = model.predict(obs, deterministic=True)
            
            # Execute action in environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Track position
            current_pos = info['current_position']
            distance = info['distance_to_goal']
            trajectory.append({
                'step': steps,
                'position': current_pos,
                'distance': distance
            })
            
            steps += 1
            
            # Optional: Slow motion for better visualization
            if slow_motion:
                time.sleep(0.05)  # 50ms delay per step
            
            # Check if episode ended
            episode_done = terminated or truncated
            
            # Print progress every 50 steps
            if steps % 50 == 0:
                print(f"    Step {steps}: Distance = {distance*1000:.2f}mm")
        
        # Episode complete - print results
        final_error = info['distance_to_goal']
        final_pos = info['current_position']
        all_errors.append(final_error)
        all_steps.append(steps)
        
        print(f"\n  ✓ Episode complete!")
        print(f"    Final position: X={final_pos[0]:.4f}m, Y={final_pos[1]:.4f}m, Z={final_pos[2]:.4f}m")
        print(f"    Steps taken: {steps}")
        print(f"    Final error: {final_error*1000:.3f}mm")
        
        # Determine score based on client requirements
        if final_error < 0.001:
            score = 8
            rating = "EXCELLENT (< 1mm)"
        elif final_error < 0.005:
            score = 6
            rating = "VERY GOOD (1-5mm)"
        elif final_error < 0.010:
            score = 4
            rating = "GOOD (5-10mm)"
        else:
            score = 0
            rating = "NEEDS IMPROVEMENT (≥ 10mm)"
        
        print(f"    Score: {score}/8 points - {rating}")
        
        if terminated:
            print(f"    Status: ✅ SUCCESS - Target reached!")
        else:
            print(f"    Status: ⏱️  TIMEOUT - Max steps reached")
        
        # Pause between episodes for viewing
        if episode < num_episodes - 1:
            print("\n  Waiting 2 seconds before next episode...")
            time.sleep(2)
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("DEMONSTRATION SUMMARY")
    print("=" * 80)
    
    mean_error = np.mean(all_errors) * 1000
    std_error = np.std(all_errors) * 1000
    min_error = np.min(all_errors) * 1000
    max_error = np.max(all_errors) * 1000
    mean_steps = np.mean(all_steps)
    
    print(f"\n📊 Statistics over {num_episodes} episodes:")
    print(f"  Average error:    {mean_error:.3f} mm")
    print(f"  Std dev error:    {std_error:.3f} mm")
    print(f"  Min error:        {min_error:.3f} mm")
    print(f"  Max error:        {max_error:.3f} mm")
    print(f"  Average steps:    {mean_steps:.1f}")
    
    # Success rates
    success_1mm = sum(1 for e in all_errors if e < 0.001) / len(all_errors) * 100
    success_5mm = sum(1 for e in all_errors if e < 0.005) / len(all_errors) * 100
    success_10mm = sum(1 for e in all_errors if e < 0.010) / len(all_errors) * 100
    
    print(f"\n🎯 Success rates:")
    print(f"  < 1mm (8 pts):    {success_1mm:.1f}%")
    print(f"  < 5mm (6 pts):    {success_5mm:.1f}%")
    print(f"  < 10mm (4 pts):   {success_10mm:.1f}%")
    
    # Average score
    scores = []
    for error in all_errors:
        if error < 0.001:
            scores.append(8)
        elif error < 0.005:
            scores.append(6)
        elif error < 0.010:
            scores.append(4)
        else:
            scores.append(0)
    avg_score = np.mean(scores)
    print(f"\n🏆 Average client score: {avg_score:.2f} / 8.0 points")
    
    print("\n" + "=" * 80)
    print("Demonstration complete! You can now close the PyBullet window.")
    print("=" * 80 + "\n")
    
    # Keep window open until user closes it
    print("PyBullet window will remain open. Close it manually when done.")
    print("Press Ctrl+C to exit this script.")
    
    try:
        # Keep script running so window stays open
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nClosing environment...")
        env.close()
        print("✓ Done!")


def demonstrate_single_episode(model_path, target_position=None, max_steps=300):
    """
    Demonstrate a single episode with optional custom target position.
    Useful for creating focused GIFs of specific scenarios.
    
    Parameters
    ----------
    model_path : str
        Path to the trained model file
    target_position : array-like, optional
        Custom target position [x, y, z]. If None, random target is used.
    max_steps : int
        Maximum steps for the episode (default: 300)
    """
    print("=" * 80)
    print("SINGLE EPISODE DEMONSTRATION")
    print("=" * 80)
    
    # Load model
    print("\nLoading model...")
    model = PPO.load(model_path)
    print("✓ Model loaded")
    
    # Create environment
    print("\nInitializing environment...")
    env = OT2Env(render=True, max_steps=max_steps, target_threshold=0.005)
    print("✓ Environment ready")
    
    # Reset environment
    obs, info = env.reset()
    
    # Override target if provided
    if target_position is not None:
        env.goal_position = np.array(target_position, dtype=np.float32)
        print(f"\n✓ Custom target set: {target_position}")
    
    goal = env.goal_position
    print(f"\n📍 Target: X={goal[0]:.4f}m, Y={goal[1]:.4f}m, Z={goal[2]:.4f}m")
    print("\nRunning episode...\n")
    
    steps = 0
    done = False
    
    while not done:
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        
        # Print progress
        if steps % 25 == 0:
            print(f"Step {steps}: Distance = {info['distance_to_goal']*1000:.2f}mm")
        
        done = terminated or truncated
    
    # Print final results
    print(f"\n✓ Episode complete!")
    print(f"  Steps: {steps}")
    print(f"  Final error: {info['distance_to_goal']*1000:.3f}mm")
    
    if terminated:
        print(f"  Status: ✅ SUCCESS")
    else:
        print(f"  Status: ⏱️  TIMEOUT")
    
    print("\n" + "=" * 80)
    print("PyBullet window will remain open. Close it manually when done.")
    print("=" * 80 + "\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nClosing...")
        env.close()


if __name__ == "__main__":
    """
    Main execution block.
    
    Choose one of the demonstration modes below.
    """
    
    # Model configuration
    MODEL_PATH = "models/260114.2210_maksym_steshkin_lr3e-4_b64_s2048_th5mm.zip"
    
    # Mode 1: Multiple episodes demonstration (recommended for GIF creation)
    print("\n" + "=" * 80)
    print("CHOOSE DEMONSTRATION MODE")
    print("=" * 80)
    print("\n1. Multiple Episodes (5 episodes with different targets)")
    print("2. Single Episode (focused demonstration)")
    print("3. Single Episode with Custom Target")
    
    choice = input("\nEnter choice (1/2/3) or press Enter for default [1]: ").strip()
    
    if choice == "2":
        # Mode 2: Single episode
        print("\n" + "=" * 80)
        demonstrate_single_episode(MODEL_PATH)
    
    elif choice == "3":
        # Mode 3: Single episode with custom target
        print("\n" + "=" * 80)
        print("Enter custom target position (or press Enter for random)")
        x = input("X position [-0.187 to 0.253] (m): ").strip()
        y = input("Y position [-0.171 to 0.220] (m): ").strip()
        z = input("Z position [0.170 to 0.289] (m): ").strip()
        
        if x and y and z:
            target = [float(x), float(y), float(z)]
            demonstrate_single_episode(MODEL_PATH, target_position=target)
        else:
            demonstrate_single_episode(MODEL_PATH)
    
    else:
        # Mode 1: Multiple episodes (default)
        print("\n" + "=" * 80)
        slow = input("Enable slow motion? (y/n) [n]: ").strip().lower() == 'y'
        demonstrate_agent(
            model_path=MODEL_PATH,
            num_episodes=5,
            slow_motion=slow,
            save_gif=False
        )
