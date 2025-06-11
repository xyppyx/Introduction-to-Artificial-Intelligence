import os
import numpy as np
import matplotlib.pyplot as plt
from flappy_bird import FlappyBirdEnv
from dqn_agent import DQNAgent
import torch
from tqdm import tqdm
import time
import random

def train_dqn(env, agent, num_episodes=2000, max_steps=10000, render_interval=50, save_interval=100):
    """
    训练DQN智能体（改进版本）
    
    参数:
    - env: 游戏环境
    - agent: DQN智能体
    - num_episodes: 训练的总回合数
    - max_steps: 每回合的最大步数
    - render_interval: 多少回合渲染一次游戏画面
    - save_interval: 多少回合保存一次模型
    """
    # 创建保存目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # 训练统计
    episode_rewards = []
    episode_lengths = []
    best_reward = -float('inf')
    
    # 学习率调度
    initial_lr = 0.0005  # 降低初始学习率
    min_lr = 0.00001
    
    # 梯度裁剪值
    grad_clip = 1.0
    
    # 奖励缩放和裁剪
    reward_scale = 0.1  # 缩小奖励以防止Q值爆炸
    
    # 定义一个递增的epsilon衰减率，开始训练时慢慢衰减
    epsilon_decay_schedule = np.linspace(0.999, 0.995, num_episodes // 4)
    epsilon_decay_idx = 0
    
    # 训练循环
    for episode in tqdm(range(1, num_episodes + 1), desc="Training"):
        # 重置环境
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        action_counts = {0: 0, 1: 0}  # 记录本回合动作统计
        
        # 设置是否渲染本回合
        render_this_episode = (episode % render_interval == 0)
        if render_this_episode:
            env_render = FlappyBirdEnv(render_mode="human")
            state, _ = env_render.reset()
            print(f"\nEpisode {episode} - Rendering")
        
        # 单回合循环
        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state)
            action_counts[action] += 1
            
            # 执行动作
            if render_this_episode:
                next_state, reward, terminated, truncated, _ = env_render.step(action)
                env_render.render()
                time.sleep(0.01)
            else:
                next_state, reward, terminated, truncated, _ = env.step(action)
            
            # 缩放奖励以防止Q值爆炸
            scaled_reward = reward * reward_scale
            
            # 改进奖励函数
            if terminated:
                # 碰撞惩罚保持不变
                scaled_reward = -1.0  # 简化碰撞惩罚
            else:
                # 存活奖励，随着时间增加而增加
                survival_bonus = 0.01 * (1 + steps / 100)
                scaled_reward += survival_bonus
                
                # 穿过管道奖励保持不变
                if reward > 0:
                    scaled_reward = 1.0  # 简化通过管道奖励
            
            # 存储经验
            agent.remember(state, action, scaled_reward, next_state, terminated)
            
            # 更频繁地训练网络
            if len(agent.replay_buffer) > agent.batch_size:
                for _ in range(4):  # 每步训练多次
                    loss = agent.train()
                    
                    # 检查是否有梯度爆炸
                    if loss > 100:
                        print(f"警告: 损失值过高 ({loss:.2f})，可能存在梯度爆炸")
            
            # 更新状态和统计
            state = next_state
            total_reward += reward  # 使用原始奖励计算总奖励
            steps += 1
            
            # 检查回合是否结束
            if terminated or truncated:
                break
        
        # 动态调整epsilon衰减率
        if episode <= num_episodes // 4:
            agent.epsilon_decay = epsilon_decay_schedule[epsilon_decay_idx]
            epsilon_decay_idx = min(epsilon_decay_idx + 1, len(epsilon_decay_schedule) - 1)
        
        # 更新探索率
        agent.update_epsilon()
        
        # 动态调整学习率
        if episode % 100 == 0:
            current_lr = max(initial_lr * (0.95 ** (episode // 100)), min_lr)
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = current_lr
                
        # 记录本回合统计
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # 每回合结束后输出动作分布
        if episode % 10 == 0 or render_this_episode:
            total_actions = sum(action_counts.values())
            action_0_pct = action_counts[0] / total_actions * 100 if total_actions > 0 else 0
            action_1_pct = action_counts[1] / total_actions * 100 if total_actions > 0 else 0
            
            print(f"Episode {episode}/{num_episodes} - "
                  f"Reward: {total_reward:.2f}, Steps: {steps}, "
                  f"Epsilon: {agent.epsilon:.4f}, "
                  f"Actions: 不跳={action_0_pct:.1f}%, 跳={action_1_pct:.1f}%")
            
            # 如果发现严重的动作不平衡，调整探索率
            if action_0_pct > 95 or action_1_pct > 95:
                if agent.epsilon < 0.5:
                    old_epsilon = agent.epsilon
                    agent.epsilon = min(0.5, agent.epsilon * 1.5)  # 增加探索
                    print(f"检测到动作不平衡，调整epsilon: {old_epsilon:.4f} -> {agent.epsilon:.4f}")
        
        # 定期检查Q值范围，防止发散
        if episode % 50 == 0:
            with torch.no_grad():
                test_states = [
                    np.array([env.screen_height/2, 0, 100, env.screen_height/2], dtype=np.float32),
                    np.array([env.screen_height/4, -5, 50, env.screen_height/3], dtype=np.float32),
                    np.array([3*env.screen_height/4, 5, 200, 2*env.screen_height/3], dtype=np.float32)
                ]
                max_q_value = 0
                for state in test_states:
                    q_values = agent.q_network(torch.FloatTensor(state).unsqueeze(0))
                    max_q = q_values.max().item()
                    max_q_value = max(max_q_value, max_q)
                
                print(f"当前最大Q值: {max_q_value:.2f}")
                
                # 如果Q值过大，重置目标网络
                if max_q_value > 1000:
                    print("Q值过大，重置目标网络")
                    agent.target_network.load_state_dict(agent.q_network.state_dict())
        
        # 保存最佳模型和定期检查点
        if total_reward > best_reward:
            best_reward = total_reward
            # 确保保存的模型格式与PyTorch 2.6兼容
            checkpoint = {
                'q_network_state_dict': agent.q_network.state_dict(),
                'target_network_state_dict': agent.target_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': float(agent.epsilon)  # 确保使用原生Python类型而不是NumPy类型
            }
            torch.save(checkpoint, 'models/best_model.pth')
            print(f"保存最佳模型，奖励: {best_reward:.2f}")
            
        if episode % save_interval == 0:
            # 确保保存的模型格式与PyTorch 2.6兼容
            checkpoint = {
                'q_network_state_dict': agent.q_network.state_dict(),
                'target_network_state_dict': agent.target_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': float(agent.epsilon)  # 确保使用原生Python类型而不是NumPy类型
            }
            torch.save(checkpoint, f'models/model_episode_{episode}.pth')
            
            # 绘制并保存训练曲线
            plot_training_results(agent, episode_rewards, episode_lengths, episode)
            
            # 定期评估模型
            if episode % (save_interval * 2) == 0:
                evaluate(agent, env, num_episodes=3, render=False)
    
    # 关闭环境
    env.close()
    if 'env_render' in locals():
        env_render.close()
    
    # 保存最终模型
    checkpoint = {
        'q_network_state_dict': agent.q_network.state_dict(),
        'target_network_state_dict': agent.target_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': float(agent.epsilon)  # 确保使用原生Python类型而不是NumPy类型
    }
    torch.save(checkpoint, 'models/final_model.pth')
    
    # 返回训练统计
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'losses': agent.losses,
        'epsilons': agent.epsilons
    }

def evaluate(agent, env, num_episodes=5, render=False):
    """在训练过程中评估模型性能"""
    rewards = []
    steps_list = []
    action_counts = {0: 0, 1: 0}
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            action_counts[action] += 1
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            state = next_state
            done = terminated or truncated
            
            if render:
                env.render()
        
        rewards.append(total_reward)
        steps_list.append(steps)
    
    avg_reward = np.mean(rewards)
    avg_steps = np.mean(steps_list)
    action_0_pct = action_counts[0] / sum(action_counts.values()) * 100
    action_1_pct = action_counts[1] / sum(action_counts.values()) * 100
    
    print(f"\n评估结果 ({num_episodes}回合):")
    print(f"平均奖励: {avg_reward:.2f}, 平均步数: {avg_steps:.2f}")
    print(f"动作分布: 不跳={action_0_pct:.1f}%, 跳={action_1_pct:.1f}%\n")
    
    return avg_reward

def plot_training_results(agent, episode_rewards, episode_lengths, episode_num):
    """绘制训练结果"""
    plt.figure(figsize=(15, 12))
    
    # 奖励曲线
    plt.subplot(3, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # 移动平均奖励
    window_size = min(50, len(episode_rewards))
    if window_size > 0:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(episode_rewards)), moving_avg, 'r-')
    
    # 回合长度
    plt.subplot(3, 2, 2)
    plt.plot(episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    # 损失曲线
    plt.subplot(3, 2, 3)
    if agent.losses:
        # 限制显示最近的损失值，防止早期大值影响图表
        recent_losses = agent.losses[-min(5000, len(agent.losses)):]
        # 将异常值（超过平均值5倍的值）替换为平均值
        mean_loss = np.mean(recent_losses)
        capped_losses = [min(l, mean_loss*5) for l in recent_losses]
        plt.plot(capped_losses)
        plt.title(f'Training Loss (Recent, Mean: {mean_loss:.4f})')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
    
    # 探索率变化
    plt.subplot(3, 2, 4)
    plt.plot(agent.epsilons)
    plt.title('Exploration Rate (Epsilon)')
    plt.xlabel('Training Step')
    plt.ylabel('Epsilon')
    
    # Q值变化（如果有记录）
    if hasattr(agent, 'q_values_record') and agent.q_values_record:
        plt.subplot(3, 2, 5)
        q_values = np.array(agent.q_values_record)
        plt.plot(q_values[:, 0], label='Action 0')
        plt.plot(q_values[:, 1], label='Action 1')
        plt.title('Average Q Values')
        plt.xlabel('Record Step')
        plt.ylabel('Q Value')
        plt.legend()
    
    # 最近回合的奖励分布
    plt.subplot(3, 2, 6)
    recent_rewards = episode_rewards[-min(100, len(episode_rewards)):]
    plt.hist(recent_rewards, bins=20)
    plt.title('Recent Rewards Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(f'plots/training_episode_{episode_num}.png')
    plt.close()

def main():
    # 设置随机种子以提高可重复性
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建环境
    env = FlappyBirdEnv()
    
    # 获取状态和动作空间维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 创建DQN智能体（使用改进的参数）
    agent = DQNAgent(
        state_dim=state_dim, 
        action_dim=action_dim,
        learning_rate=0.0001,  # 降低学习率
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.999,  # 更慢的衰减
        buffer_size=100000,   # 更大的缓冲区
        batch_size=64,        # 更大的批量
        target_update=5       # 更频繁的目标网络更新
    )
    
    # 添加Q值记录属性
    agent.q_values_record = []
    
    # 训练智能体
    train_stats = train_dqn(
        env=env,
        agent=agent,
        num_episodes=5000,    # 更多回合
        max_steps=5000,
        render_interval=100,
        save_interval=50
    )
    
    # 绘制最终训练结果
    plot_training_results(agent, train_stats['episode_rewards'], train_stats['episode_lengths'], "final")
    
    print("Training completed!")

if __name__ == "__main__":
    main()
