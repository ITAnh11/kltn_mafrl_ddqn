import numpy as np
import matplotlib.pyplot as plt
from uav_mec_env import UAVMECEnv
from parameters import *
from marl.marl import MARL
from mafrl.mafrl import MAFRL
from marl.marl_ddqn import MARL_DDQN
from mafrl.mafrl_ddqn import MAFRL_DDQN

# === Thông số mô phỏng ===
num_steps = 500  # số time slot mô phỏng mỗi case (hoặc 1000 tuỳ máy)
latency_list = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
]  # độ trễ từ 0.1s đến 1.0s


SEED = 42
type_ue = 100

# Các list lưu average power với từng số lượng UE
greedy_results = []
random_results = []
local_results = []
marl_results = []
marl_ddqn_results = []
mafrl_results = []
mafrl_ddqn_results = []


for max_latency in latency_list:  # số UAV cố định
    print(f"Đang chạy với độ trễ {max_latency}s...")

    # Khởi tạo env với số UE cụ thể
    env = UAVMECEnv(num_ues=100, max_latency=max_latency)

    greedy_power, random_power, local_power = [], [], []

    # === Greedy Offload ===
    print("Bắt đầu Greedy Offload...")
    env.reset(seed=SEED)
    for _ in range(num_steps):
        actions = env.greedy_offloading()
        _, reward, terminated, _, info = env.step(actions)
        greedy_power.append(info["total_energy_consumption"])
        if terminated:
            env.reset()

    # === Random Offload ===
    print("Bắt đầu Random Offload...")
    env.reset(seed=SEED)
    for _ in range(num_steps):
        actions = env.random_offloading()
        _, reward, terminated, _, info = env.step(actions)
        random_power.append(info["total_energy_consumption"])
        if terminated:
            env.reset()

    # # === Local Execute ===
    print("Bắt đầu Local Execution...")
    env.reset(seed=SEED)
    for _ in range(num_steps):
        actions = env.local_execution()
        _, reward, terminated, _, info = env.step(actions)
        local_power.append(info["total_energy_consumption"])
        if terminated:
            env.reset()

    print("Bắt đầu MARL...")
    marl = MARL(num_ues=100, num_uavs=10, num_episodes=25000, max_latency=max_latency)
    marl.env.reset(seed=SEED)
    marl.load_model(f"marl/model/marl_model_ver_ue_{type_ue}.pth")
    average_energy_marl = marl.test(num_steps=num_steps)

    print("Bắt đầu MAFRL...")
    mafrl = MAFRL(num_ues=100, num_uavs=10, num_episodes=25000, max_latency=max_latency)
    mafrl.env.reset(seed=SEED)
    mafrl.load_model(f"mafrl/model/mafrl_model_ver_ue_{type_ue}_v1.pth")
    average_energy_mafrl = mafrl.test(num_steps=num_steps)

    print("Bắt đầu MAFRL-DDQN...")
    mafrl_ddqn = MAFRL_DDQN(
        num_ues=100, num_uavs=10, num_episodes=25000, max_latency=max_latency
    )
    mafrl_ddqn.env.reset(seed=SEED)
    mafrl_ddqn.load_model(f"mafrl/model/mafrl_ddqn_model_ver_ue_{type_ue}_v1.pth")
    average_energy_mafrl_ddqn = mafrl_ddqn.test(num_steps=num_steps)

    # Lưu trung bình power từng baseline
    greedy_results.append(np.sum(greedy_power))
    random_results.append(np.sum(random_power))
    local_results.append(np.sum(local_power))
    marl_results.append(average_energy_marl * num_steps)
    mafrl_results.append(average_energy_mafrl * num_steps)
    mafrl_ddqn_results.append(average_energy_mafrl_ddqn * num_steps)

# === Vẽ biểu đồ như Fig.3 ===
plt.figure(figsize=(10, 6))
plt.plot(latency_list, greedy_results, marker="^", label="Greedy Offload", linewidth=2)
plt.plot(latency_list, random_results, marker="d", label="Random Execute", linewidth=2)
plt.plot(latency_list, local_results, marker="s", label="Local Execute", linewidth=2)
plt.plot(latency_list, marl_results, marker="o", label="MARL", linewidth=2)
plt.plot(latency_list, mafrl_results, marker="x", label="MAFRL", linewidth=2)
plt.plot(latency_list, mafrl_ddqn_results, marker="v", label="MAFRL-DDQN", linewidth=2)

plt.xlabel("Latency (s)", fontsize=13)
plt.ylabel("Sum Energy Consumption (J)", fontsize=13)
plt.title("Impact of Latency on Sum Energy Consumption", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig("all_vs_latency.png")
