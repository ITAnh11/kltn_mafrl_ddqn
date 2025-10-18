from mafrl.mafrl import MAFRL
from mafrl.mafrl_ddqn import MAFRL_DDQN

mafrl_ddqn = MAFRL_DDQN(num_ues=20, num_uavs=10, num_episodes=30000)
mafrl_ddqn.run(save_csv=True, save_csv_file="results/MAFRL_DDQN_reward_20.csv")
mafrl_ddqn.save_model("mafrl/model/mafrl_ddqn_model_ver_ue_20_v2.pth")


# mafrl = MAFRL(num_ues=100, num_uavs=10, num_episodes=30000)
# mafrl.run(save_csv=True)
# mafrl.save_model("mafrl/model/mafrl_model_ver_ue_100_v2.pth")
