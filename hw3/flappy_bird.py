import pygame
import random
import numpy as np
import os
from pygame.locals import *
import gymnasium as gym
from gymnasium import spaces

class FlappyBirdEnv(gym.Env):
    """
    Flappy Bird环境
    实现了gymnasium接口，可以用于强化学习训练
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, render_mode=None, difficulty='easy'):
        super(FlappyBirdEnv, self).__init__()
        
        # 初始化pygame
        pygame.init()
        
        # 环境参数设置
        self.screen_width = 288
        self.screen_height = 512
        
        # 根据难度调整参数
        if difficulty == 'easy':
            self.pipe_gap = 160       # 更大的通过空间
            self.gravity = 0.8        # 较小的重力
            self.bird_flap_acc = -8   # 适中的跳跃力度
            self.pipe_speed = 2       # 更慢的管道速度
        else:  # 'normal' 或其他
            self.pipe_gap = 100
            self.gravity = 1
            self.bird_flap_acc = -9
            self.pipe_speed = 3
        
        self.pipe_width = 52
        self.bird_width = 34
        self.bird_height = 24
        self.ground_height = 100
        
        # 定义动作空间：0-不动，1-跳跃
        self.action_space = spaces.Discrete(2)
        
        # 定义观察空间：[鸟的高度, 鸟的速度, 到下一个管道的水平距离, 下一个管道的高度]
        self.observation_space = spaces.Box(
            low=np.array([0, -10, 0, 0]),
            high=np.array([self.screen_height, 10, self.screen_width, self.screen_height]),
            dtype=np.float32
        )
        
        # 设置渲染模式
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.isopen = True
        
        # 加载游戏资源
        self._load_resources()
        
        # 初始化游戏状态
        self.reset()
    
    def _load_resources(self):
        """加载游戏所需的图片资源"""
        # 创建资源目录
        os.makedirs('assets', exist_ok=True)
        
        # 使用pygame内置资源或创建简单图形
        self.background = pygame.Surface((self.screen_width, self.screen_height))
        self.background.fill((135, 206, 250))  # 天蓝色背景
        
        self.bird_surface = pygame.Surface((self.bird_width, self.bird_height), pygame.SRCALPHA)
        pygame.draw.ellipse(self.bird_surface, (255, 255, 0), (0, 0, self.bird_width, self.bird_height))  # 黄色小鸟
        
        self.pipe_surface = pygame.Surface((self.pipe_width, self.screen_height), pygame.SRCALPHA)
        pygame.draw.rect(self.pipe_surface, (0, 128, 0), (0, 0, self.pipe_width, self.screen_height))  # 绿色管道
        
        self.ground_surface = pygame.Surface((self.screen_width, self.ground_height))
        self.ground_surface.fill((222, 184, 135))  # 棕色地面
    
    def reset(self, seed=None, options=None):
        """重置环境状态"""
        super().reset(seed=seed)
        
        # 初始化小鸟位置和速度
        self.bird_pos = [self.screen_width // 3, self.screen_height // 2]
        self.bird_vel = 0
        
        # 初始化管道
        self.pipes = []
        self._generate_pipe()
        
        # 游戏状态
        self.score = 0
        self.ticks = 0
        self.game_active = True
        
        # 获取初始观察
        observation = self._get_observation()
        info = {}
        
        # 初始化渲染组件
        if self.render_mode == "human" and self.screen is None:
            pygame.display.init()
            pygame.display.set_caption('Flappy Bird')
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()
        
        return observation, info
    
    def _generate_pipe(self):
        """生成新的管道"""
        pipe_height = random.randint(50, self.screen_height - self.ground_height - self.pipe_gap - 50)
        pipe_x = self.screen_width + 10
        
        self.pipes.append({
            'x': pipe_x,
            'height': pipe_height
        })
    
    def _get_observation(self):
        """获取当前状态的观察向量"""
        if not self.pipes:
            # 如果没有管道，创建默认观察值
            return np.array([
                self.bird_pos[1],
                self.bird_vel,
                self.screen_width,
                self.screen_height // 2
            ], dtype=np.float32)
        
        # 找到最近的管道
        next_pipe = None
        for pipe in self.pipes:
            if pipe['x'] + self.pipe_width > self.bird_pos[0]:
                next_pipe = pipe
                break
        
        if next_pipe is None:
            next_pipe = self.pipes[0]
        
        # 计算到下一个管道的水平距离
        horizontal_dist = next_pipe['x'] + self.pipe_width - self.bird_pos[0]
        if horizontal_dist < 0:
            horizontal_dist = self.screen_width  # 如果已经过了所有管道，设置一个默认值
        
        return np.array([
            self.bird_pos[1],
            self.bird_vel,
            horizontal_dist,
            next_pipe['height']
        ], dtype=np.float32)
    
    def step(self, action):
        """执行动作，更新环境状态"""
        reward = 0.1  # 存活奖励
        terminated = False
        truncated = False
        
        # 执行动作
        if action == 1:  # 跳跃
            self.bird_vel = self.bird_flap_acc
        
        # 更新小鸟位置
        self.bird_vel += self.gravity
        self.bird_pos[1] += self.bird_vel
        
        # 检查碰撞
        if self._check_collision():
            reward = -10  # 碰撞惩罚
            terminated = True
        
        # 更新管道位置
        for pipe in self.pipes:
            pipe['x'] -= self.pipe_speed
        
        # 移除屏幕外的管道
        if self.pipes and self.pipes[0]['x'] + self.pipe_width < 0:
            self.pipes.pop(0)
            reward += 1  # 通过管道奖励
            self.score += 1
        
        # 根据需要生成新管道
        if not self.pipes or self.pipes[-1]['x'] < self.screen_width - 150:
            self._generate_pipe()
        
        # 更新游戏时钟
        self.ticks += 1
        
        # 获取新的观察
        observation = self._get_observation()
        
        # 信息字典
        info = {
            'score': self.score
        }
        
        return observation, reward, terminated, truncated, info
    
    def _check_collision(self):
        """检查小鸟是否发生碰撞"""
        # 检查是否撞到地面或天花板
        if self.bird_pos[1] <= 0 or self.bird_pos[1] >= self.screen_height - self.ground_height:
            return True
        
        # 检查是否撞到管道
        for pipe in self.pipes:
            # 检查x轴碰撞
            if self.bird_pos[0] + self.bird_width > pipe['x'] and self.bird_pos[0] < pipe['x'] + self.pipe_width:
                # 检查y轴碰撞 (上管道和下管道)
                if self.bird_pos[1] < pipe['height'] or self.bird_pos[1] + self.bird_height > pipe['height'] + self.pipe_gap:
                    return True
        
        return False
    
    def render(self):
        """渲染当前环境状态"""
        if self.render_mode is None:
            return
        
        if self.screen is None and self.render_mode == "human":
            pygame.display.init()
            pygame.display.set_caption('Flappy Bird')
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()
        
        if self.screen is None and self.render_mode == "rgb_array":
            pygame.display.init()
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()
        
        # 绘制背景
        self.screen.blit(self.background, (0, 0))
        
        # 绘制管道
        for pipe in self.pipes:
            # 上管道
            top_pipe = pygame.transform.flip(self.pipe_surface, False, True)
            top_pipe = pygame.transform.scale(top_pipe, (self.pipe_width, pipe['height']))
            self.screen.blit(top_pipe, (pipe['x'], 0))
            
            # 下管道
            bottom_pipe = pygame.transform.scale(
                self.pipe_surface, 
                (self.pipe_width, self.screen_height - pipe['height'] - self.pipe_gap)
            )
            self.screen.blit(bottom_pipe, (pipe['x'], pipe['height'] + self.pipe_gap))
        
        # 绘制地面
        self.screen.blit(self.ground_surface, (0, self.screen_height - self.ground_height))
        
        # 绘制小鸟
        self.screen.blit(self.bird_surface, (self.bird_pos[0], self.bird_pos[1]))
        
        # 绘制分数
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        
        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
    
    def close(self):
        """关闭环境"""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
            self.screen = None
