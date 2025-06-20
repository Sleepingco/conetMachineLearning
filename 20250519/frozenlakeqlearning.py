# -*- coding: utf-8 -*-
"""FrozenLakeQLearning

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10fTyCLqjs9QsvaW85hyaNY-Caoub_LWq
"""

import numpy as np
import random as pr
import gymnasium as gym

# 환경 설정
env = gym.make("FrozenLake-v1",is_slippery=False)
# Q 테이블 초기화 (상태 × 행동)

Q = np.zeros((env.observation_space.n,env.action_space.n))

num_episodes = 2000
rewards_per_episode = []

for epidose in range(num_episodes):
  state,_ = env.reset()
  done = False
  total_reward = 0

  while not done:
    # Q값이 가장 큰 행동 선택
    action = np.argmax(Q[state])
    # 행동 수행 및 결과 관찰
    next_state,reward,terminated,truncated,_ = env.step(action)
    done = terminated or truncated
    # Q 테이블 업데이트 (α=1, γ=1 가정)
    Q[state, action] = reward + np.max(Q[next_state])
    # 다음 상태로 이동
    state = next_state
    total_reward += reward
  rewards_per_episode.append(total_reward)
# 결과 출력
print(f"\n last 100 episode mean reward:{np.mean(rewards_per_episode[-100:]):.4f}")
print("Lates Q table")
print(Q)

def rargmax(vector):
    """
    주어진 벡터에서 최댓값을 가지는 인덱스들 중 하나를 무작위로 반환한다.
    예: [1, 3, 3, 2] → 인덱스 1과 2 중 하나 무작위 선택
    """
    m = np.amax(vector)                           # 최대값 구하기
    indices = np.flatnonzero(vector == m)         # 최대값의 인덱스들 추출
    return pr.choice(indices)                     # 그 중 무작위 선택


# FrozenLake 환경 생성 (미끄리짐 없음: 결정론적)
env = gym.make("FrozenLake-v1",is_slippery=False)
# 상태 수 x 행동 수 크기의 Q 테이블 초기화
Q = np.zeros((env.observation_space.n,env.action_space.n))
# 에피소드 수 설정
num_episodes = 2000

# 에피 소드 별 총 보상을 저장할 리스트
rewards_per_episode = []

for epidose in range(num_episodes):
  state,_ = env.reset()
  done = False
  total_reward = 0

  while not done:
    # Q값이 가장 큰 행동 선택
    action = rargmax(Q[state])
    # 행동 수행 및 결과 관찰
    next_state,reward,terminated,truncated,_ = env.step(action)
    done = terminated or truncated
    # Q 테이블 업데이트 (α=1, γ=1 가정)
    Q[state, action] = reward + np.max(Q[next_state])
    # 다음 상태로 이동
    state = next_state
    total_reward += reward
  rewards_per_episode.append(total_reward)
# 결과 출력
print(f"\n last 100 episode mean reward:{np.mean(rewards_per_episode[-100:]):.4f}")
print("Lates Q table")
print("[좌,하,우,상]")
print(Q)

def rargmax(vector):
    """
    주어진 벡터에서 최댓값을 가지는 인덱스들 중 하나를 무작위로 반환한다.
    예: [1, 3, 3, 2] → 인덱스 1과 2 중 하나 무작위 선택
    """
    m = np.amax(vector)                           # 최대값 구하기
    indices = np.flatnonzero(vector == m)         # 최대값의 인덱스들 추출
    return pr.choice(indices)                     # 그 중 무작위 선택

rewards_per_episode = []
total_reward = 0
for episode in range(num_episodes):
  print(episode)
  state,_ = env.reset()
  done = False
  total_reward = 0

  while not done:
    # 현재 상태에서 Q값이 가장 큰 행동을 선택 (동일하면 무작위)
    action = rargmax(Q[state])
    # 선택한 행동을 환경에 적용하여 다음 상태, 보상, 종료 여부 획득
    next_state,reward,terminated,truncated,_ = env.step(action)
    done = terminated or truncated
    # Q 테이블 업데이트 (학습률, 할인율 없이 단순 갱신)
    Q[state,action] = reward + np.max(Q[next_state])
    # 상태 업데이트

    state = next_state
    total_reward += reward
  # 한 에피소드에서 받은 총 보상 기록
  rewards_per_episode.append(total_reward)
  print(rewards_per_episode)

import numpy as np
import gymnasium as gym
import random as pr
import matplotlib.pyplot as plt
# rargmax 함수: Q값이 동일한 경우 무작위 선택
def rargmax(vector):
  m = np.amax(vector)
  indices = np.nonzero(vector == m)[0]
  return pr.choice(indices)
# 환경 생성
env = gym.make("FrozenLake-v1", is_slippery=False)
# Q 테이블 초기화 (상태 수 × 행동 수)
Q = np.zeros([env.observation_space.n, env.action_space.n])
# 에피소드 수
num_episodes = 2000
# 각 에피소드에서 받은 총 보상 저장 리스트
rList = []
for episode in range(num_episodes):
  state, _ = env.reset()
  done = False
  total_reward = 0
  while not done:
    action = rargmax(Q[state])
    # action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))
    # action = rargmax(Q[state])
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    Q[state, action] = reward + np.max(Q[next_state])
    state = next_state
    total_reward += reward
  rList.append(total_reward)
# 출력
print("Success rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
# 시각화: 에피소드별 보상 히스토그램
plt.bar(range(len(rList)), rList, color="blue")
plt.title("Rewards per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

from math import trunc
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import random as pr

def ragmax(vector):
  m=np.amax(vector)
  indices = np.nonzero(vector == m)[0]
  return pr.choice(indices)
# create env
evn = gym.make("FrozenLake-v1",map_name='4x4',is_slippery=False)

# initilalize table with zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
# Discount factor
dis = .999
num_episodes = 2000
# Create list to contain total rewars and steps per episode
rList = []

for i in range(num_episodes):
  # Reset enviorment and get first new observation (Gymnasium 방식으로 수정)
  state,_ = env.reset()
  rAll = 0
  done = False

  # The Q-Table learning algorithm
  while not done:
    # Choose an action by greedily (with noise) picking from Q table
    action = np.argmax(Q[state,:]+np.random.randn(1,env.action_space.n)/(i+1))

    # Get new state and reward from enviorment (Gymnasium 방식으로 수정)
    new_state, reward,terminated,truncated,_ = env.step(action)
    done = terminated or truncated

    # Update Q-Table with new knowledge using decay rate
    Q[state, action] = reward + dis * np.max(Q[new_state,:])

    rAll += reward
    state = new_state
  rList.append(rAll)
print("Success rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
# 시각화: 에피소드별 보상 히스토그램
plt.bar(range(len(rList)), rList, color="blue")
plt.show()

import gymnasium as gym

# create env
env = gym.make("FrozenLake-v1",map_name="4x4",is_slippery=True)

# reset
state, info = env.reset()
# Macros
LEFT =0
DOWN=1
RIGHT=2
UP=3
# KeyboardMappring
arrow_keys={
    'w':UP,
    's':DOWN,
    'd':RIGHT,
    'a':LEFT
}

# 상태 출력 함수(4x4 grid)
def print_state(state):
  state_desc = env.unwrapped.desc.tolist()
  state_desc = [[c.decode('utf-8')for c in line] for line in state_desc]
  row,col = divmod(state,4)
  state_desc[row][col] = 'P' # current loc
  for line in state_desc:
    print("".join(line))
  print()
print("Use 'w','a','s','d' keys to move the agent.")
print('Press "q" tp quit the game. \n')

print("initial state")
print_state(state)

while True:
  key = input("Enter action (w/a/s/d): ").strip().lower()
  if key == 'q':
    print("Game aborted")
    break
  if key not in arrow_keys:
    print('invaild key. Use "w","a","s","d"')
    continue

  action = arrow_keys[key]
  next_state,reward,terminated,truncated,info =  env.step(action)

  # 결과 출력
  action_names = {
                    UP:'UP',
                    DOWN:'DOWN',
                    LEFT:'LEFT',
                    RIGHT:"RIGHT",
                  }
  print(f"\nAction:{action_names[action]},State:{next_state},Reward:{reward},Info{info}")

  print("Current state")
  print_state(next_state)
  state = next_state

  if terminated or truncated:
      if reward == 1.0:
        print("Congratulations! You reached the goal!")
      else:
        print("Game over. You fell into a hole.")
      break
print("Game ended")

import numpy as np
import gymnasium as gym # Gymnasium 라이브러리 사용
import matplotlib.pyplot as plt
import random as pr
def rargmax(vector):
  """최대 Q값 중 무작위 인덱스 선택"""
  m = np.amax(vector)
  indices = np.nonzero(vector == m)[0]
  return pr.choice(indices)
# FrozenLake 환경 생성 (Gymnasium 방식)
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
# Q 테이블 초기화
Q = np.zeros([env.observation_space.n, env.action_space.n])
# 할인율 및 학습률 설정
dis = 0.99
num_episodes = 2000
learning_rate = 0.7
# 보상 리스트 초기화
rList = []
for i in range(num_episodes):
  # 환경 리셋
  state, _ = env.reset()
  rAll = 0
  done = False
  # Q-Table 학습
  while not done:
    # 노이즈 추가한 Q값으로 행동 선택
    action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))
    # 환경에서 다음 상태 및 보상 가져오기
    new_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    # Q-값 업데이트
    Q[state, action] = (1 - learning_rate) * Q[state, action] + learning_rate * (reward + dis * np.max(Q[new_state, :]))
    rAll += reward
    state = new_state
  rList.append(rAll)
# 결과 출력 (소수점 3자리로 제한)
print("Success rate: {:.3f}".format(sum(rList) / num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(np.round(Q, 3))
plt.bar(range(len(rList)), rList, color="blue")
plt.show()

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import random as pr

def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

# 실험 파라미터
discount_factors = [0.9, 0.95, 0.99]
learning_rates = [0.1, 0.5, 0.7]
num_episodes = 2000

results = {}

for dis in discount_factors:
    for learning_rate in learning_rates:
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
        Q = np.zeros([env.observation_space.n, env.action_space.n])
        rList = []

        for i in range(num_episodes):
            state, _ = env.reset()
            rAll = 0
            done = False

            while not done:
                action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))
                new_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                Q[state, action] = (1 - learning_rate) * Q[state, action] + \
                                   learning_rate * (reward + dis * np.max(Q[new_state, :]))
                rAll += reward
                state = new_state

            rList.append(rAll)

        avg_reward = np.mean(rList)
        results[(dis, learning_rate)] = avg_reward
        print(f"할인율={dis}, 학습률={learning_rate}, 평균 성공률={avg_reward:.4f}")

# 시각화
import seaborn as sns
import pandas as pd

df = pd.DataFrame([
    {"할인율 γ": dis, "학습률 α": lr, "성공률": results[(dis, lr)]}
    for dis in discount_factors
    for lr in learning_rates
])
# pivot = df.pivot("할인율 γ", "학습률 α", "성공률")

# plt.figure(figsize=(8, 5))
# sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlG