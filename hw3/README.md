# Flappy Bird DQN 强化学习项目

本项目使用深度强化学习方法（DQN及其变体）训练AI智能体玩Flappy Bird游戏。通过结合多种先进的强化学习技术，如Double DQN和优先经验回放，实现了有效的游戏策略学习。

## 环境要求

- Python 3.11
- PyTorch 2.7.0
- Pygame 2.6.1
- NumPy 2.2.6
- Matplotlib 3.10.3
- OpenCV-Python 4.11.0.86
- tqdm 4.67.1

## 快速开始

1. **安装依赖**

```bash
pip install torch numpy pygame matplotlib opencv-python tqdm gymnasium
```

2. **训练模型**

```bash
python train.py
```

3. **测试模型**

```bash
python test.py
```

## 项目结构

- `flappy_bird.py`: 基于Pygame的Flappy Bird游戏环境，遵循Gymnasium接口标准
- `dqn_agent.py`: DQN智能体实现，包含多种强化学习算法变体
- `train.py`: 基础训练脚本
- `test.py`: 模型测试和评估脚本
- `convert_model.py`: 模型格式转换工具（适用于PyTorch 2.6+）
- `models/`: 保存训练好的模型
- `plots/`: 训练过程可视化图表
- `videos/`: 测试过程录制的视频

## 功能特性

### 1. 强化学习算法

本项目实现了多种先进的深度强化学习技术：

- **DQN (Deep Q-Network)**: 基础Q学习算法与深度神经网络结合
- **Double DQN**: 解决Q值过估计问题
- **优先经验回放 (PER)**: 基于TD误差的样本重要性加权

### 2. 环境定制

- 可调节难度的Flappy Bird环境
- 灵活的奖励函数设计
- 完整的可视化支持

### 3. 训练优化

- 动态学习率调度
- 自适应探索策略
- 梯度裁剪防止发散
- 模型异常检测与重置

## 调参指南

以下参数可以在训练脚本中调整以获得更好的性能：

- **学习率 (learning_rate)**: 控制模型参数更新步长，建议范围 0.00001~0.0005
- **探索率衰减 (epsilon_decay)**: 控制随机探索减少速度，建议值 0.9995
- **经验缓冲区大小 (buffer_size)**: 更大的缓冲区有助于稳定学习，建议值 100000+
- **批量大小 (batch_size)**: 每次更新使用的样本数，建议值 64~256
- **目标网络更新频率 (target_update)**: 建议值 5~10
- **奖励缩放**: 可以调整奖励函数以引导更好的学习

## 模型训练分析

在训练过程中，您可能会观察到以下现象：

1. **初始阶段**: 智能体主要依靠随机探索，表现不佳
2. **中期阶段**: 开始学习基本生存策略，但可能表现不稳定
3. **后期阶段**: 逐渐掌握穿越管道的技巧，分数稳步提高

如果训练出现问题（如Q值爆炸、行为退化），可以使用以下策略：

- 降低学习率
- 增加梯度裁剪阈值
- 重置网络权重
- 调整奖励函数

## 故障排除

1. **模型加载错误**: 如果使用PyTorch 2.6+遇到加载问题，请运行 `convert_model.py`转换模型格式
2. **性能不佳**: 尝试增加训练回合数或调整 `train.py`中的超参数
3. **Q值爆炸**: 查看测试输出中的"当前最大Q值"，如果超过100，可能需要降低学习率或增加梯度裁剪
4. **渲染问题**: 如遇Pygame渲染错误，可能需要更新显示驱动或调整环境配置
