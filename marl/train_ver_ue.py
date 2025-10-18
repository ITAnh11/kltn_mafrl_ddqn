from marl.marl import MARL

marl = MARL(num_ues=20, num_uavs=10, num_episodes=30000)
marl.run(save_csv=True, save_csv_file="results/MARL_reward_20.csv")
marl.save_model("marl/model/marl_ddqn_model_ver_ue_20_v2.pth")
