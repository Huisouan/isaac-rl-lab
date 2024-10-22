# Isaac Lab VQVAE

本仓库仅用于存储代码。VQVAE目前可以进行学习，但是无法收敛。还望各位大佬指点。

## Installation

本人只在Ubuntu24.04上进行验证安装，但win上面的安装应当是大同小异的。

安装可以参考
[https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html](https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html)
也可以使用如下代码对isaacsim进行安装
Conda环境：

```
conda create -n isaaclab python=3.10
conda activate isaaclab
```

venv:

```
# create a virtual environment named isaaclab with python3.10
python3.10 -m venv isaaclab
# activate the virtual environment
source isaaclab/bin/activate
```

下一步，安装所需的python包

```
#如果你的显卡支持CUDA11，清使用cu118
#pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade pip
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk isaacsim-extscache-kit isaacsim-app --extra-index-url https://pypi.nvidia.com
#下面一行可以用于验证isaacsim的安装是否成功了，当然这并不是必须的
# note: you can pass the argument "--help" to see all arguments possible.
isaacsim
```

下面我们可以安装这个代码库里面的内容了：

```
git clone git@github.com:Huisouan/isaac-lab-vqvae.git
cd isaac-lab-vqvae
```

如果你是ubuntu系统，安装如下

```
sudo apt install cmake build-essential
./isaaclab.sh --install # or "./isaaclab.sh -i"
```

如果你是windows

```
isaaclab.bat --install :: or "isaaclab.bat -i"
```

安装完之后会有一个用于eula，直接yes就可以

### ！下载模型数据到本地使用！

由于nvidia的usd模型都是在neclus服务器上面的，因此每次我们要使用一个模型的时候，都会从nvidia的服务器上下载，这非常耗费时间，因此我们可以选择将模型寻址的路径改成本地。

首先我们需要将asset下载下来，我们可以到nvidia-omniverse官网来下载omniverse-launcher[官网入口](https://www.nvidia.cn/omniverse/)
有win和linux两个版本的，根据自己情况下载就行。
安装好之后就是这个页面：
![1729590467093](images/README/1729590467093.png)

资产的下载和安装可以按照这个说明来进行下载：
[https://docs-prod.omniverse.nvidia.com/isaacsim/latest/installation/install_faq.html#isaac-sim-setup-nucleus-add-assets-mount](https://docs-prod.omniverse.nvidia.com/isaacsim/latest/installation/install_faq.html#isaac-sim-setup-nucleus-add-assets-mount)

在资产下载好之后，寻找到

```
source/extensions/omni.isaac.lab/omni/isaac/lab/utils/assets.py
```

这个文件，把里面的NUCLEUS_ASSET_ROOT_DIR 改成本地的文件夹路径，就可以实现本地的文件寻址

## How to use

### train

直接运行`source/vqvae/train.py`就可以进行训练，如果想要在训练时进行渲染，则将`setattr(args_cli, 'headless', True)`这一行注释掉就可以。

### play

运行`source/vqvae/play.py`就可以播放训练结果。

## main content

### 强化学习的环境位置:

```
source/vqvae/vqvae/tasks/direct/GO2/go2_pmc_env.py
```

其中：PMCEnvCfg是环境的配置参数，PMCEnv是强化学习环境本体。PMCEnv类里面的step函数是仿真流程。
仿真环境初始化，reward，get_dones,apply_action等都在这里面。

数据集使用MotionData类导入，在`source/vqvae/vqvae/datasets/motionload.py`文件里面。
数据文件保存在`source/vqvae/data`里面，目前这个数据还有一点问题，但应该不怎么影响训练效果。
如果你想要看一下数据集是什么样的，可以运行

```
source/vqvae/playdatasets.py
```

来播放一下目前有的数据集

### PPO算法以及网络loss计算的位置：

```
source/vqvae/vqvae/rsl_rl/pmc_algorithm/pmcppo.py
```

主要包括了ppo算法里面的policy，value 以及vqvae的loss的计算，以及网络的反向传播

### pmc 网络

```
source/vqvae/vqvae/rsl_rl/modules/pmc.py
```

这一部分包含了pmc网络的init部分以及forward部分，froward部分的算法在函数`update_distribution`里面实现。

