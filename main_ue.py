from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from uav_mec_env import UAVMECEnv
from parameters import *
from marl.marl import MARL
from mafrl.mafrl_base import MAFRL
from marl.marl_ddqn import MARL_DDQN
from mafrl.mafrl_ddqn import MAFRL_DDQN

from common import CSVLogger

# === Thông số mô phỏng ===
num_steps = 200  # số time slot mô phỏng mỗi case (hoặc 1000 tuỳ máy)
ue_list = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]  # số UE giống paper
# ue_list = [60, 80, 100, 120, 140, 160]  # số UE giống paper
# ue_list = [20, 40, 60, 80, 100]  # số UE giống paper

SEED = 42
type_ue = 100

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = f"results/results_ue"
os.makedirs(outdir, exist_ok=True)

csv_logger = CSVLogger(
    os.path.join(outdir, "results_ue.csv"),
    fieldnames=[
        "num_ues",
        "policy",
        "reward",
        "total_power_system",
        "total_power_ue",
        "total_power_uav",
        "violations",
        "vio_local",
        "vio_offload",
    ],
)

for num_ue in ue_list:  # số UAV cố định
    print(f"Đang chạy với {num_ue} UE...")

    # Khởi tạo env với số UE cụ thể
    env = UAVMECEnv(num_ues=num_ue)

    avg_power_system = 0.0
    avg_power_ue = 0.0
    avg_power_uav = 0.0
    avg_violations = 0.0
    avg_vio_local = 0.0
    avg_vio_offload = 0.0
    avg_reward = 0.0

    # === Greedy Offload ===
    print("Bắt đầu Greedy Offload...")
    env.reset(seed=SEED)
    for _ in range(num_steps):
        actions = env.greedy_offloading()
        _, reward, terminated, _, info = env.step(actions)
        avg_power_system += info["total_power_system"]
        avg_power_ue += info["total_power_ue"]
        avg_power_uav += info["total_power_uav"]
        avg_violations += info["violations"]
        avg_vio_local += info["vio_local"]
        avg_vio_offload += info["vio_offload"]
        avg_reward += info["reward"]

    csv_logger.log(
        {
            "num_ues": num_ue,
            "policy": "Greedy Offload",
            "reward": np.round(avg_reward / num_steps, 6),
            "total_power_system": np.round(avg_power_system / num_steps, 6),
            "total_power_ue": np.round(avg_power_ue / num_steps, 6),
            "total_power_uav": np.round(avg_power_uav / num_steps, 6),
            "violations": np.round(avg_violations / num_steps, 6),
            "vio_local": np.round(avg_vio_local / num_steps, 6),
            "vio_offload": np.round(avg_vio_offload / num_steps, 6),
        }
    )

    avg_power_system = 0.0
    avg_power_ue = 0.0
    avg_power_uav = 0.0
    avg_violations = 0.0
    avg_vio_local = 0.0
    avg_vio_offload = 0.0
    avg_reward = 0.0

    # === Random Offload ===
    print("Bắt đầu Random Offload...")
    env.reset(seed=SEED)
    for _ in range(num_steps):
        actions = env.random_offloading()
        _, reward, terminated, _, info = env.step(actions)
        avg_power_system += info["total_power_system"]
        avg_power_ue += info["total_power_ue"]
        avg_power_uav += info["total_power_uav"]
        avg_violations += info["violations"]
        avg_vio_local += info["vio_local"]
        avg_vio_offload += info["vio_offload"]
        avg_reward += info["reward"]

    csv_logger.log(
        {
            "num_ues": num_ue,
            "policy": "Random Offload",
            "reward": np.round(avg_reward / num_steps, 6),
            "total_power_system": np.round(avg_power_system / num_steps, 6),
            "total_power_ue": np.round(avg_power_ue / num_steps, 6),
            "total_power_uav": np.round(avg_power_uav / num_steps, 6),
            "violations": np.round(avg_violations / num_steps, 6),
            "vio_local": np.round(avg_vio_local / num_steps, 6),
            "vio_offload": np.round(avg_vio_offload / num_steps, 6),
        }
    )

    avg_power_system = 0.0
    avg_power_ue = 0.0
    avg_power_uav = 0.0
    avg_violations = 0.0
    avg_vio_local = 0.0
    avg_vio_offload = 0.0
    avg_reward = 0.0

    # # === Local Execute ===
    print("Bắt đầu Local Execution...")
    env.reset(seed=SEED)
    for _ in range(num_steps):
        actions = env.local_execution()
        _, reward, terminated, _, info = env.step(actions)
        avg_power_system += info["total_power_system"]
        avg_power_ue += info["total_power_ue"]
        avg_power_uav += info["total_power_uav"]
        avg_violations += info["violations"]
        avg_vio_local += info["vio_local"]
        avg_vio_offload += info["vio_offload"]
        avg_reward += info["reward"]

    csv_logger.log(
        {
            "num_ues": num_ue,
            "policy": "Local Execution",
            "reward": avg_reward / num_steps,
            "total_power_system": np.round(avg_power_system / num_steps, 6),
            "total_power_ue": np.round(avg_power_ue / num_steps, 6),
            "total_power_uav": np.round(avg_power_uav / num_steps, 6),
            "violations": np.round(avg_violations / num_steps, 6),
            "vio_local": np.round(avg_vio_local / num_steps, 6),
            "vio_offload": np.round(avg_vio_offload / num_steps, 6),
        }
    )

    # print("Bắt đầu MARL...")
    # avg_power_system = 0.0
    # avg_power_ue = 0.0
    # avg_power_uav = 0.0
    # avg_violations = 0.0
    # avg_vio_local = 0.0
    # avg_vio_offload = 0.0
    # avg_reward = 0.0

    # marl = MARL(num_ues=num_ue, num_uavs=10)
    # marl.env.reset(seed=SEED)
    # marl.load_model(f"marl/model/marl_model_ue_100_20251119_232329.pth")
    # average_marl = marl.test(num_steps=num_steps)

    # csv_logger.log(
    #     {
    #         "num_ues": num_ue,
    #         "policy": "MARL",
    #         "reward": np.round(average_marl["avg_reward"], 6),
    #         "total_power_system": np.round(average_marl["avg_power_system"], 6),
    #         "total_power_ue": np.round(average_marl["avg_power_ue"], 6),
    #         "total_power_uav": np.round(average_marl["avg_power_uav"], 6),
    #         "violations": np.round(average_marl["avg_violations"], 6),
    #         "vio_local": np.round(average_marl["avg_vio_local"], 6),
    #         "vio_offload": np.round(average_marl["avg_vio_offload"], 6),
    #     }
    # )

    print("Bắt đầu MAFRL...")
    avg_power_system = 0.0
    avg_power_ue = 0.0
    avg_power_uav = 0.0
    avg_violations = 0.0
    avg_vio_local = 0.0
    avg_vio_offload = 0.0
    avg_reward = 0.0

    mafrl = MAFRL(num_ues=num_ue, num_uavs=10)
    mafrl.env.reset(seed=SEED)
    # Tải mô hình đã train trước đó
    mafrl.load_model(f"mafrl/model/mafrl_model_ue_100_20251120_021325.pth")
    average_mafrl = mafrl.test(num_steps=num_steps)
    csv_logger.log(
        {
            "num_ues": num_ue,
            "policy": "MAFRL",
            "reward": np.round(average_mafrl["avg_reward"], 6),
            "total_power_system": np.round(average_mafrl["avg_power_system"], 6),
            "total_power_ue": np.round(average_mafrl["avg_power_ue"], 6),
            "total_power_uav": np.round(average_mafrl["avg_power_uav"], 6),
            "violations": np.round(average_mafrl["avg_violations"], 6),
            "vio_local": np.round(average_mafrl["avg_vio_local"], 6),
            "vio_offload": np.round(average_mafrl["avg_vio_offload"], 6),
        }
    )

    avg_power_system = 0.0
    avg_power_ue = 0.0
    avg_power_uav = 0.0
    avg_violations = 0.0
    avg_vio_local = 0.0
    avg_vio_offload = 0.0
    avg_reward = 0.0

    print("Bắt đầu MAFRL-DDQN...")
    mafrl_ddqn = MAFRL_DDQN(num_ues=num_ue, num_uavs=10)
    mafrl_ddqn.env.reset(seed=SEED)
    # Tải mô hình đã train trước đó
    mafrl_ddqn.load_model(f"mafrl/model/mafrl_ddqn_model_ue_100_20251120_002322.pth")
    average_mafrl_ddqn = mafrl_ddqn.test(num_steps=num_steps)
    csv_logger.log(
        {
            "num_ues": num_ue,
            "policy": "MAFRL-DDQN",
            "reward": np.round(average_mafrl_ddqn["avg_reward"], 6),
            "total_power_system": np.round(average_mafrl_ddqn["avg_power_system"], 6),
            "total_power_ue": np.round(average_mafrl_ddqn["avg_power_ue"], 6),
            "total_power_uav": np.round(average_mafrl_ddqn["avg_power_uav"], 6),
            "violations": np.round(average_mafrl_ddqn["avg_violations"], 6),
            "vio_local": np.round(average_mafrl_ddqn["avg_vio_local"], 6),
            "vio_offload": np.round(average_mafrl_ddqn["avg_vio_offload"], 6),
        }
    )

csv_logger.close()
