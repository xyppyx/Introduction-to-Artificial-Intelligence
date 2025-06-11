import os
import time
import numpy as np
import torch
from flappy_bird import FlappyBirdEnv
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

def test_agent(model_path, num_episodes=10, render=True, record=False, debug=True):
    """
    测试训练好的智能体
    
    参数:
    - model_path: 加载模型的路径
    - num_episodes: 测试的回合数
    - render: 是否渲染游戏画面
    - record: 是否录制视频
    - debug: 是否输出调试信息
    """
    # 创建环境
    render_mode = "human" if render else None
    env = FlappyBirdEnv(render_mode=render_mode)
    
    # 获取状态和动作空间维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 创建智能体并加载模型
    agent = DQNAgent(state_dim, action_dim)
    agent.load(model_path)
    print(f"Model loaded from {model_path}")
    
    # 检查模型是否正确加载
    if debug:
        # 检查Q网络的状态
        print("\n--- 模型诊断信息 ---")
        print(f"Q网络结构: {agent.q_network}")
        
        # 获取模型的参数统计
        total_params = sum(p.numel() for p in agent.q_network.parameters())
        trainable_params = sum(p.numel() for p in agent.q_network.parameters() if p.requires_grad)
        print(f"总参数数量: {total_params}, 可训练参数: {trainable_params}")
        
        # 检查模型权重是否有合理值
        for name, param in agent.q_network.named_parameters():
            if param.requires_grad:
                print(f"{name} - 平均值: {param.data.mean().item():.6f}, 标准差: {param.data.std().item():.6f}")
        
        # 检查epsilon值
        print(f"Epsilon值: {agent.epsilon:.6f}")
        
        # 测试几个随机状态的Q值分布
        print("\n--- Q值测试 ---")
        test_states = [
            np.array([env.screen_height/2, 0, 100, env.screen_height/2], dtype=np.float32),  # 静止鸟在中间，管道100距离
            np.array([env.screen_height/4, -5, 50, env.screen_height/3], dtype=np.float32),  # 上升的鸟，管道近
            np.array([3*env.screen_height/4, 5, 200, 2*env.screen_height/3], dtype=np.float32)  # 下降的鸟，管道远
        ]
        
        with torch.no_grad():
            for i, state in enumerate(test_states):
                q_values = agent.q_network(torch.FloatTensor(state).unsqueeze(0))
                action = torch.argmax(q_values).item()
                print(f"测试状态 {i+1}: Q值 = {q_values.numpy()}, 选择动作 = {action}")
    
    # 视频录制设置
    video_writer = None
    if record:
        os.makedirs('videos', exist_ok=True)
        video_path = f'videos/flappy_bird_test_{int(time.time())}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (env.screen_width, env.screen_height))
        print(f"Recording video to {video_path}")
    
    # 测试统计
    episode_rewards = []
    episode_lengths = []
    action_counts = {0: 0, 1: 0}  # 统计选择的动作分布
    
    # 测试循环
    for episode in tqdm(range(num_episodes), desc="Testing"):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        episode_actions = {0: 0, 1: 0}  # 本回合动作统计
        
        if debug:
            print(f"\n--- 回合 {episode+1} 初始状态: {state} ---")
        
        while not done:
            # 智能体选择动作（不使用探索）
            action = agent.select_action(state, training=False)
            
            # 记录动作统计
            action_counts[action] += 1
            episode_actions[action] += 1
            
            if debug and steps % 5 == 0:  # 每5步输出一次状态和动作信息
                print(f"Step {steps}: 状态={state}, 动作={action}")
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # 渲染环境
            if render:
                frame = env.render()
                
                # 录制视频
                if record and video_writer is not None:
                    if frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        video_writer.write(frame)
                
                time.sleep(0.02)  # 减慢渲染速度
            
            # 更新状态和统计
            state = next_state
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
            if debug and terminated:
                print(f"回合结束原因: 碰撞, 当前状态: {state}")
        
        # 记录本回合统计
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {episode+1}/{num_episodes}, Score: {total_reward:.2f}, Steps: {steps}, 动作分布: 不跳={episode_actions[0]}, 跳={episode_actions[1]}")
    
    # 关闭视频写入器
    if video_writer is not None:
        video_writer.release()
    
    # 关闭环境
    env.close()
    
    # 显示测试统计
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    print(f"\nTest Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Episode Length: {avg_length:.2f}")
    print(f"总动作分布: 不跳={action_counts[0]} ({action_counts[0]/sum(action_counts.values())*100:.1f}%), "
          f"跳={action_counts[1]} ({action_counts[1]/sum(action_counts.values())*100:.1f}%)")
    
    # 绘制测试结果
    plt.figure(figsize=(15, 10))
    
    # 奖励和步数
    plt.subplot(2, 2, 1)
    plt.bar(range(num_episodes), episode_rewards)
    plt.title('Test Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(2, 2, 2)
    plt.bar(range(num_episodes), episode_lengths)
    plt.title('Test Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    # 动作分布饼图
    plt.subplot(2, 2, 3)
    plt.pie([action_counts[0], action_counts[1]], 
            labels=['No Jump', 'Jump'], 
            autopct='%1.1f%%',
            startangle=90)
    plt.title('Action Distribution')
    
    # 添加一个分析图表（例如，奖励随时间变化）
    if len(episode_rewards) > 1:
        plt.subplot(2, 2, 4)
        cumulative_rewards = np.cumsum(episode_rewards)
        plt.plot(cumulative_rewards)
        plt.title('Cumulative Reward')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
    
    plt.tight_layout()
    plt.savefig('plots/test_results.png')
    plt.show()
    
    # 提供可能的问题诊断
    if avg_reward < 0:
        print("\n--- 性能问题诊断 ---")
        if action_counts[0] == 0 or action_counts[1] == 0:
            print("问题: 智能体只执行一种动作，这表明网络可能没有正确学习或参数设置有问题。")
            print("建议: 检查训练过程，增加训练时间，或调整奖励函数。")
        
        if avg_length < 50:
            print("问题: 回合长度过短，智能体可能没有学会基本的生存技能。")
            print("建议: 可能需要更简单的初始环境，或者调整训练参数使学习更稳定。")
        
        print("\n可能的解决方案:")
        print("1. 增加训练时间/回合数")
        print("2. 调整学习率")
        print("3. 修改网络结构")
        print("4. 重新设计奖励函数，例如添加更多存活奖励")
        print("5. 实现课程学习，从简单场景逐渐增加难度")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'action_counts': action_counts
    }

def main():
    # 测试最佳模型
    model_path = 'models/best_model.pth'
    if not os.path.exists(model_path):
        model_path = 'models/final_model.pth'
    
    if not os.path.exists(model_path):
        print("No trained model found. Please run train.py first.")
        return
    
    test_agent(
        model_path=model_path,
        num_episodes=5,
        render=True,
        record=True,
        debug=True  # 启用调试模式
    )

if __name__ == "__main__":
    main()
