"""
RL Model Evaluation Script for OT-2 Robot Control

This script loads a trained reinforcement learning model and evaluates its performance
on the OT-2 robot positioning task. It provides comprehensive metrics including:
- Positioning accuracy (error in mm)
- Success rate at different thresholds (1mm, 5mm, 10mm)
- Episode completion time
- Comparison with random baseline
- Performance visualizations

The script fulfills the client requirements by measuring positioning error and
awarding points based on the accuracy criteria:
    - < 1mm: 8 points
    - 1-5mm: 6 points
    - 5-10mm: 4 points
    - ≥ 10mm: 0 points

Usage:
    python test_rl.py

Requirements:
    - stable-baselines3
    - gymnasium
    - numpy
    - matplotlib
    - sim_class.py (OT-2 simulation)
    - maksym_steshkin_ot2_gym_wrapper.py (custom environment)

Author: Maksym Steshkin
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from maksym_steshkin_ot2_gym_wrapper import OT2Env
import time
from pathlib import Path


class RLEvaluator:
    """
    Evaluator for trained RL models on OT-2 positioning task.
    
    Tracks metrics, generates visualizations, and compares performance
    against client requirements and baseline controllers.
    
    Parameters
    ----------
    model_path : str
        Path to the trained model file
    num_episodes : int
        Number of evaluation episodes (default: 100)
    render : bool
        Whether to render simulation visually (default: False)
    """
    
    def __init__(self, model_path, num_episodes=100, render=False):
        self.model_path = model_path
        self.num_episodes = num_episodes
        self.render = render
        
        # Metrics storage
        self.final_errors = []
        self.episode_lengths = []
        self.episode_rewards = []
        self.success_times = []
        self.trajectories = []
        
        # Client requirement thresholds (in meters)
        self.threshold_1mm = 0.001
        self.threshold_5mm = 0.005
        self.threshold_10mm = 0.010
        
    def load_model(self):
        """Load the trained RL model."""
        print(f"Loading model from: {self.model_path}")
        self.model = PPO.load(self.model_path)
        print("✓ Model loaded successfully")
        
    def evaluate(self):
        """
        Run evaluation episodes and collect metrics.
        
        Returns
        -------
        dict
            Dictionary containing all evaluation metrics
        """
        print("=" * 70)
        print("RL MODEL EVALUATION")
        print("=" * 70)
        
        # Load model
        self.load_model()
        
        # Create environment
        print(f"\nInitializing environment...")
        env = OT2Env(render=self.render, max_steps=300, target_threshold=0.005)
        print(f"✓ Environment initialized")
        
        print(f"\nRunning {self.num_episodes} evaluation episodes...")
        print("-" * 70)
        
        # Run evaluation episodes
        for episode in range(self.num_episodes):
            obs, info = env.reset()
            episode_reward = 0.0
            episode_done = False
            steps = 0
            trajectory = []
            
            while not episode_done:
                # Get action from trained model
                action, _states = self.model.predict(obs, deterministic=True)
                
                # Execute action
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Track metrics
                episode_reward += reward
                steps += 1
                trajectory.append({
                    'step': steps,
                    'position': info['current_position'],
                    'goal': info['goal_position'],
                    'distance': info['distance_to_goal']
                })
                
                # Check if episode ended
                episode_done = terminated or truncated
                
                # Record success time if goal reached
                if terminated and len(self.success_times) == episode:
                    self.success_times.append(steps)
            
            # Store episode metrics
            final_error = info['distance_to_goal']
            self.final_errors.append(final_error)
            self.episode_lengths.append(steps)
            self.episode_rewards.append(episode_reward)
            self.trajectories.append(trajectory)
            
            # Progress reporting
            if (episode + 1) % 10 == 0:
                avg_error = np.mean(self.final_errors) * 1000
                success_rate = self._calculate_success_rate(0.005)
                print(f"Episode {episode + 1}/{self.num_episodes} | "
                      f"Avg error: {avg_error:.2f}mm | "
                      f"Success rate: {success_rate:.1f}%")
        
        env.close()
        print("-" * 70)
        
        # Calculate and return metrics
        return self._compute_metrics()
    
    def _calculate_success_rate(self, threshold):
        """Calculate success rate for a given threshold."""
        successes = sum(1 for error in self.final_errors if error < threshold)
        return (successes / len(self.final_errors)) * 100 if self.final_errors else 0.0
    
    def _compute_metrics(self):
        """Compute comprehensive evaluation metrics."""
        metrics = {
            # Basic statistics
            'num_episodes': self.num_episodes,
            'mean_error': np.mean(self.final_errors),
            'std_error': np.std(self.final_errors),
            'median_error': np.median(self.final_errors),
            'min_error': np.min(self.final_errors),
            'max_error': np.max(self.final_errors),
            
            # Success rates at different thresholds
            'success_rate_1mm': self._calculate_success_rate(self.threshold_1mm),
            'success_rate_5mm': self._calculate_success_rate(self.threshold_5mm),
            'success_rate_10mm': self._calculate_success_rate(self.threshold_10mm),
            
            # Episode metrics
            'mean_episode_length': np.mean(self.episode_lengths),
            'mean_reward': np.mean(self.episode_rewards),
            'mean_success_time': np.mean(self.success_times) if self.success_times else None,
            
            # Client scoring
            'client_score': self._calculate_client_score(),
            
            # Raw data
            'final_errors': self.final_errors,
            'episode_lengths': self.episode_lengths,
            'episode_rewards': self.episode_rewards
        }
        
        return metrics
    
    def _calculate_client_score(self):
        """
        Calculate score based on client requirements.
        
        Scoring:
        - < 1mm: 8 points
        - 1-5mm: 6 points
        - 5-10mm: 4 points
        - ≥ 10mm: 0 points
        """
        scores = []
        for error in self.final_errors:
            if error < self.threshold_1mm:
                scores.append(8)
            elif error < self.threshold_5mm:
                scores.append(6)
            elif error < self.threshold_10mm:
                scores.append(4)
            else:
                scores.append(0)
        
        return np.mean(scores)
    
    def print_results(self, metrics):
        """Print formatted evaluation results."""
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        print("\n📊 POSITIONING ACCURACY:")
        print(f"  Mean error:   {metrics['mean_error']*1000:.3f} mm")
        print(f"  Std dev:      {metrics['std_error']*1000:.3f} mm")
        print(f"  Median error: {metrics['median_error']*1000:.3f} mm")
        print(f"  Min error:    {metrics['min_error']*1000:.3f} mm")
        print(f"  Max error:    {metrics['max_error']*1000:.3f} mm")
        
        print("\n🎯 SUCCESS RATES:")
        print(f"  < 1mm (8 pts):   {metrics['success_rate_1mm']:.1f}%")
        print(f"  < 5mm (6 pts):   {metrics['success_rate_5mm']:.1f}%")
        print(f"  < 10mm (4 pts):  {metrics['success_rate_10mm']:.1f}%")
        
        print("\n⚡ EFFICIENCY:")
        print(f"  Avg episode length: {metrics['mean_episode_length']:.1f} steps")
        print(f"  Avg reward:         {metrics['mean_reward']:.2f}")
        if metrics['mean_success_time']:
            print(f"  Avg success time:   {metrics['mean_success_time']:.1f} steps")
        
        print("\n🏆 CLIENT SCORE:")
        print(f"  Average score: {metrics['client_score']:.2f} / 8.0 points")
        
        # Performance rating
        if metrics['client_score'] >= 7.5:
            rating = "EXCELLENT ⭐⭐⭐"
        elif metrics['client_score'] >= 6.5:
            rating = "VERY GOOD ⭐⭐"
        elif metrics['client_score'] >= 5.0:
            rating = "GOOD ⭐"
        else:
            rating = "NEEDS IMPROVEMENT"
        print(f"  Rating: {rating}")
        
        print("=" * 70)
    
    def plot_results(self, metrics, save_path="evaluation_results.png"):
        """
        Create comprehensive visualization of evaluation results.
        
        Parameters
        ----------
        metrics : dict
            Metrics dictionary from evaluation
        save_path : str
            Path to save the figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('RL Model Evaluation Results - OT-2 Positioning Task', 
                     fontsize=16, weight='bold')
        
        # 1. Error distribution histogram
        ax = axes[0, 0]
        errors_mm = np.array(metrics['final_errors']) * 1000
        ax.hist(errors_mm, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(1, color='green', linestyle='--', label='1mm (8pts)')
        ax.axvline(5, color='orange', linestyle='--', label='5mm (6pts)')
        ax.axvline(10, color='red', linestyle='--', label='10mm (4pts)')
        ax.set_xlabel('Final Positioning Error (mm)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Error Distribution', fontsize=12, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Success rates bar chart
        ax = axes[0, 1]
        thresholds = ['< 1mm\n(8pts)', '< 5mm\n(6pts)', '< 10mm\n(4pts)']
        rates = [metrics['success_rate_1mm'], 
                 metrics['success_rate_5mm'], 
                 metrics['success_rate_10mm']]
        colors = ['green', 'orange', 'red']
        bars = ax.bar(thresholds, rates, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Success Rate (%)', fontsize=11)
        ax.set_title('Success Rates by Threshold', fontsize=12, weight='bold')
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, weight='bold')
        
        # 3. Episode lengths over time
        ax = axes[0, 2]
        ax.plot(metrics['episode_lengths'], color='purple', alpha=0.6, linewidth=1)
        ax.axhline(np.mean(metrics['episode_lengths']), color='red', 
                  linestyle='--', label=f'Mean: {np.mean(metrics["episode_lengths"]):.1f}')
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Steps to Completion', fontsize=11)
        ax.set_title('Episode Length Over Time', fontsize=12, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Error over episodes
        ax = axes[1, 0]
        errors_mm = np.array(metrics['final_errors']) * 1000
        ax.plot(errors_mm, color='coral', alpha=0.6, linewidth=1)
        ax.axhline(1, color='green', linestyle='--', alpha=0.5, label='1mm')
        ax.axhline(5, color='orange', linestyle='--', alpha=0.5, label='5mm')
        ax.axhline(10, color='red', linestyle='--', alpha=0.5, label='10mm')
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Final Error (mm)', fontsize=11)
        ax.set_title('Positioning Error Over Episodes', fontsize=12, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Cumulative error distribution
        ax = axes[1, 1]
        sorted_errors = np.sort(errors_mm)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        ax.plot(sorted_errors, cumulative, color='darkblue', linewidth=2)
        ax.axvline(1, color='green', linestyle='--', alpha=0.5, label='1mm')
        ax.axvline(5, color='orange', linestyle='--', alpha=0.5, label='5mm')
        ax.axvline(10, color='red', linestyle='--', alpha=0.5, label='10mm')
        ax.set_xlabel('Final Error (mm)', fontsize=11)
        ax.set_ylabel('Cumulative Percentage (%)', fontsize=11)
        ax.set_title('Cumulative Error Distribution', fontsize=12, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Client score breakdown
        ax = axes[1, 2]
        score_labels = ['8 pts\n(<1mm)', '6 pts\n(1-5mm)', '4 pts\n(5-10mm)', '0 pts\n(≥10mm)']
        score_counts = [
            sum(1 for e in metrics['final_errors'] if e < 0.001),
            sum(1 for e in metrics['final_errors'] if 0.001 <= e < 0.005),
            sum(1 for e in metrics['final_errors'] if 0.005 <= e < 0.010),
            sum(1 for e in metrics['final_errors'] if e >= 0.010)
        ]
        colors_score = ['green', 'orange', 'gold', 'red']
        wedges, texts, autotexts = ax.pie(score_counts, labels=score_labels, colors=colors_score,
                                           autopct='%1.1f%%', startangle=90, 
                                           textprops={'fontsize': 10, 'weight': 'bold'})
        ax.set_title(f'Client Score Distribution\nAvg: {metrics["client_score"]:.2f}/8.0', 
                    fontsize=12, weight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {save_path}")
        
        return fig
    
    def plot_sample_trajectories(self, num_samples=4, save_path="trajectories.png"):
        """
        Plot sample trajectories showing robot path to goal.
        
        Parameters
        ----------
        num_samples : int
            Number of sample trajectories to plot
        save_path : str
            Path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
        
        fig.suptitle('Sample Episode Trajectories', fontsize=14, weight='bold')
        
        # Select diverse episodes (best, worst, median, random)
        errors_with_idx = [(i, e) for i, e in enumerate(self.final_errors)]
        errors_with_idx.sort(key=lambda x: x[1])
        
        selected_indices = [
            errors_with_idx[0][0],  # Best
            errors_with_idx[len(errors_with_idx)//2][0],  # Median
            errors_with_idx[-1][0],  # Worst
            np.random.randint(0, len(self.trajectories))  # Random
        ]
        
        titles = ['Best Performance', 'Median Performance', 'Worst Performance', 'Random Sample']
        
        for idx, (traj_idx, title) in enumerate(zip(selected_indices, titles)):
            if idx >= num_samples:
                break
                
            ax = axes[idx]
            traj = self.trajectories[traj_idx]
            
            # Extract positions
            positions = np.array([step['position'] for step in traj])
            goal = traj[0]['goal']
            final_error = traj[-1]['distance']
            
            # Plot trajectory in 3D projection
            ax.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.6, linewidth=2, label='Path')
            ax.scatter(positions[0, 0], positions[0, 1], c='green', s=150, 
                      marker='o', edgecolors='black', linewidths=2, label='Start', zorder=5)
            ax.scatter(goal[0], goal[1], c='red', s=150, 
                      marker='*', edgecolors='black', linewidths=2, label='Goal', zorder=5)
            ax.scatter(positions[-1, 0], positions[-1, 1], c='blue', s=100, 
                      marker='x', linewidths=3, label='Final', zorder=5)
            
            ax.set_xlabel('X Position (m)', fontsize=10)
            ax.set_ylabel('Y Position (m)', fontsize=10)
            ax.set_title(f'{title}\nError: {final_error*1000:.2f}mm, Steps: {len(traj)}', 
                        fontsize=11, weight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Trajectories saved to: {save_path}")
        
        return fig


def main():
    """
    Main execution function for RL model evaluation.
    """
    # Configuration
    MODEL_PATH = "models/260114.2210_maksym_steshkin_lr3e-4_b64_s2048_th5mm.zip"
    NUM_EPISODES = 100
    RENDER = False
    
    print("\n" + "=" * 70)
    print("OT-2 RL CONTROLLER EVALUATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Episodes: {NUM_EPISODES}")
    print(f"  Render: {RENDER}")
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"\n❌ Error: Model not found at {MODEL_PATH}")
        print("Please ensure the trained model file exists.")
        return
    
    # Create evaluator
    evaluator = RLEvaluator(
        model_path=MODEL_PATH,
        num_episodes=NUM_EPISODES,
        render=RENDER
    )
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    # Print results
    evaluator.print_results(metrics)
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    evaluator.plot_results(metrics, save_path="maksym_steshkin_evaluation_results.png")
    evaluator.plot_sample_trajectories(num_samples=4, save_path="maksym_steshkin_trajectories.png")
    
    print("\n✅ Evaluation complete!")
    print("\nGenerated files:")
    print("  - maksym_steshkin_evaluation_results.png (metrics visualization)")
    print("  - maksym_steshkin_trajectories.png (sample trajectories)")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
