from rl_lab.assets.loder_for_algs import VQVAEMotion

if __name__ == "__main__":
    # 初始化 VQVAEMotion 类
    vqvae_motion = VQVAEMotion(data_dir="datasets/mocap_motions_go2", datatype="isaacgym", file_type="txt")
    # 准备数据
    vqvae_motion.prepare_vqvae_state_trans()
    # 打印验证信息
    print("state shape:", vqvae_motion.state.shape)
    print("state_003 shape:", vqvae_motion.state_003.shape)
    print("state_006 shape:", vqvae_motion.state_006.shape)
    print("state_03 shape:", vqvae_motion.state_03.shape)