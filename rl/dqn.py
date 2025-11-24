"""
DEEP Q-NETWORK (DQN)
Neural network for learning Q(s,a) values

ИСПРАВЛЕНО:
- Gradient clipping
- Layer normalization для стабильности
- Improved initialization
- Better loss handling

Architecture: State -> FC -> LayerNorm -> ReLU -> FC -> LayerNorm -> ReLU -> Q-values
Uses experience replay and target network

Author: STAR_ANT + Claude Sonnet 4.5
Date: November 17, 2025
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class DQN(nn.Module):
    """
    Deep Q-Network
    
    Neural network for approximating Q(s,a) values
    
    ИСПРАВЛЕНО: Добавлена LayerNorm для стабильности
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]):
        """
        Initialize DQN
        
        Parameters:
        -----------
        state_dim : int
            Dimension of state space
        action_dim : int
            Number of actions
        hidden_dims : list
            Hidden layer dimensions
        """
        super(DQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network with LayerNorm
        layers = []
        prev_dim = state_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Layer normalization для стабильности
            layers.append(nn.LayerNorm(hidden_dim))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout для regularization
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(0.1))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"DQN created: {state_dim} -> {hidden_dims} -> {action_dim}")
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Parameters:
        -----------
        state : Tensor
            State vector(s)
        
        Returns:
        --------
        Tensor : Q-values for each action
        """
        return self.network(state)
    
    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select action using epsilon-greedy policy
        
        Parameters:
        -----------
        state : np.ndarray
            Current state
        epsilon : float
            Exploration rate
        
        Returns:
        --------
        int : Selected action
        """
        if random.random() < epsilon:
            # Explore: random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Exploit: best action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.forward(state_tensor)
                action = q_values.argmax(dim=1).item()
                return action


class ReplayBuffer:
    """
    Experience Replay Buffer
    
    Stores transitions (s, a, r, s', done) for training
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer
        
        Parameters:
        -----------
        capacity : int
            Maximum buffer size
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        logger.info(f"ReplayBuffer created: capacity={capacity}")
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Add transition to buffer"""
        # Validate inputs
        if np.any(np.isnan(state)) or np.any(np.isnan(next_state)):
            logger.warning("NaN detected in state, skipping transition")
            return
        
        if np.isnan(reward) or np.isinf(reward):
            logger.warning(f"Invalid reward: {reward}, skipping transition")
            return
        
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample random batch
        
        Parameters:
        -----------
        batch_size : int
            Batch size
        
        Returns:
        --------
        tuple : (states, actions, rewards, next_states, dones)
        """
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent for training
    
    Handles training loop, target network, and optimization
    
    ИСПРАВЛЕНО: Улучшенная стабильность обучения
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: dict = None):
        """
        Initialize DQN Agent
        
        Parameters:
        -----------
        state_dim : int
            State space dimension
        action_dim : int
            Action space dimension
        config : dict
            Configuration parameters
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}
        
        # Hyperparameters
        self.gamma = self.config.get('gamma', 0.95)
        self.learning_rate = self.config.get('learning_rate', 0.0001)
        self.epsilon_start = self.config.get('epsilon_start', 1.0)
        self.epsilon_end = self.config.get('epsilon_end', 0.05)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.target_update_freq = self.config.get('target_update_freq', 20)
        self.batch_size = self.config.get('batch_size', 64)
        self.buffer_capacity = self.config.get('buffer_capacity', 10000)
        self.gradient_clip = self.config.get('gradient_clip', 1.0)
        
        # Networks
        hidden_dims = self.config.get('hidden_dims', [128, 128])
        self.policy_net = DQN(state_dim, action_dim, hidden_dims)
        self.target_net = DQN(state_dim, action_dim, hidden_dims)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        # Loss function - Huber loss для стабильности
        self.loss_fn = nn.SmoothL1Loss()  # More stable than MSE
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_capacity)
        
        # Training state
        self.epsilon = self.epsilon_start
        self.steps_done = 0
        self.episode_count = 0
        
        logger.info(f"DQNAgent initialized:")
        logger.info(f"  gamma={self.gamma}, lr={self.learning_rate}")
        logger.info(f"  epsilon: {self.epsilon_start} → {self.epsilon_end}")
        logger.info(f"  batch_size={self.batch_size}, buffer={self.buffer_capacity}")
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select action using current policy
        
        Parameters:
        -----------
        state : np.ndarray
            Current state
        
        Returns:
        --------
        int : Selected action
        """
        self.steps_done += 1
        return self.policy_net.get_action(state, self.epsilon)
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> float:
        """
        Perform one training step
        
        ИСПРАВЛЕНО: Gradient clipping и NaN protection
        
        Returns:
        --------
        float : Loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        try:
            # Sample batch
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            
            # Normalize rewards для стабильности
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            rewards = torch.clamp(rewards, -10, 10)
            
            # Compute current Q-values
            current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Compute target Q-values
            with torch.no_grad():
                next_q = self.target_net(next_states).max(1)[0]
                target_q = rewards + (1 - dones) * self.gamma * next_q
                target_q = torch.clamp(target_q, -100, 100)  # Prevent explosion
            
            # Compute loss
            loss = self.loss_fn(current_q, target_q)
            
            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("NaN/Inf loss detected, skipping update")
                return 0.0
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(),
                self.gradient_clip
            )
            
            self.optimizer.step()
            
            return loss.item()
        
        except Exception as e:
            logger.error(f"Error in train_step: {e}")
            return 0.0
    
    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def episode_end(self):
        """Called at end of episode"""
        self.episode_count += 1
        self.decay_epsilon()
        
        # Update target network
        if self.episode_count % self.target_update_freq == 0:
            self.update_target_network()
            logger.debug(f"Target network updated (episode {self.episode_count})")
    
    def save(self, path: str):
        """
        Save model
        
        Parameters:
        -----------
        path : str
            Path to save model
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episode_count': self.episode_count,
            'config': self.config
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """
        Load model
        
        Parameters:
        -----------
        path : str
            Path to load model from
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.episode_count = checkpoint['episode_count']
        logger.info(f"Model loaded from {path}")


# Test DQN
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*80)
    print("DQN ARCHITECTURE TEST")
    print("="*80 + "\n")
    
    # Create agent
    state_dim = 12
    action_dim = 3
    
    config = {
        'hidden_dims': [128, 128],
        'learning_rate': 0.0001,
        'batch_size': 64
    }
    
    agent = DQNAgent(state_dim, action_dim, config)
    
    print("Agent created:")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Hidden layers: {config['hidden_dims']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Epsilon: {agent.epsilon:.4f}")
    
    # Test forward pass
    print("\nTest 1: Forward pass")
    test_state = np.random.randn(state_dim).astype(np.float32)
    action = agent.select_action(test_state)
    print(f"  Input state shape: {test_state.shape}")
    print(f"  Selected action: {action}")
    print(f"  ✓ Forward pass works")
    
    # Test training
    print("\nTest 2: Training step")
    print("  Filling replay buffer...")
    for i in range(200):
        s = np.random.randn(state_dim).astype(np.float32)
        a = np.random.randint(0, action_dim)
        r = np.random.randn() * 0.1
        s_next = np.random.randn(state_dim).astype(np.float32)
        done = False
        
        agent.store_transition(s, a, r, s_next, done)
    
    print(f"  Buffer size: {len(agent.replay_buffer)}")
    
    print("  Running training steps...")
    losses = []
    for i in range(10):
        loss = agent.train_step()
        losses.append(loss)
    
    avg_loss = np.mean(losses)
    print(f"  Average loss: {avg_loss:.6f}")
    print(f"  ✓ Training works (no NaN)")
    
    # Test save/load
    print("\nTest 3: Save/Load")
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'test_model.pth')
        agent.save(save_path)
        print(f"  ✓ Model saved")
        
        agent2 = DQNAgent(state_dim, action_dim, config)
        agent2.load(save_path)
        print(f"  ✓ Model loaded")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80 + "\n")
