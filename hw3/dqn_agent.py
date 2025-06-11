import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    """
    深度Q网络模型
    """
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        
        # 定义神经网络结构
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),  # 更大的网络
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """使用适当的初始化方法初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用He初始化，适合ReLU激活函数
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        return self.layers(x)

class ReplayBuffer:
    """
    经验回放缓冲区
    存储和采样训练数据
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """添加经验到缓冲区"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """随机采样一批经验"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为torch张量
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    DQN智能体
    实现深度Q学习算法
    """
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon_start  # 探索率
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_count = 0
        
        # 创建Q网络和目标网络
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 冻结目标网络参数
        for param in self.target_network.parameters():
            param.requires_grad = False
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # 损失函数 - 使用平滑L1损失而不是MSE，对异常值更鲁棒
        self.loss_fn = nn.SmoothL1Loss()
        
        # 训练统计
        self.losses = []
        self.rewards = []
        self.epsilons = []
        
        # Q值记录
        self.q_values_record = []
        self.record_interval = 10  # 每10次训练记录一次
        self.record_counter = 0
    
    def select_action(self, state, training=True):
        """选择动作：探索或利用"""
        if training and random.random() < self.epsilon:
            # 探索：随机选择动作
            return random.randrange(self.action_dim)
        else:
            # 利用：选择Q值最大的动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                
                # 记录Q值
                if training and self.record_counter % self.record_interval == 0:
                    self.q_values_record.append(q_values.numpy()[0])
                
                self.record_counter += 1
                return torch.argmax(q_values).item()
    
    def update_epsilon(self):
        """更新探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilons.append(self.epsilon)
    
    def train(self):
        """训练Q网络"""
        # 如果经验不足，不进行训练
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # 从经验回放中采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 计算当前Q值
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值，使用Double DQN方法
        with torch.no_grad():
            # 使用在线网络选择动作
            next_actions = self.q_network(next_states).max(1)[1].unsqueeze(1)
            # 使用目标网络评估这些动作
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            # 计算目标值
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
            # 裁剪目标值，防止极端值
            max_q = 100.0  # 设置合理的上限
            target_q_values = torch.clamp(target_q_values, -max_q, max_q)
        
        # 计算损失
        loss = self.loss_fn(q_values, target_q_values)
        
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪以防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # 定期更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 记录损失
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        self.replay_buffer.add(state, action, reward, next_state, done)
        # 记录奖励
        self.rewards.append(reward)
    
    def save(self, path):
        """保存模型"""
        # 确保所有值都是原生Python类型，避免NumPy类型导致的兼容性问题
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': float(self.epsilon)  # 确保使用原生Python类型
        }
        torch.save(checkpoint, path)

    def load(self, path):
        """加载模型"""
        try:
            # 尝试使用新的默认参数加载
            checkpoint = torch.load(path)
        except Exception as e:
            print(f"使用默认设置加载失败，尝试使用weights_only=False: {e}")
            # 如果失败，使用weights_only=False重试
            checkpoint = torch.load(path, weights_only=False)
            print("成功加载模型文件")
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = float(checkpoint['epsilon'])  # 确保转换为原生Python类型
