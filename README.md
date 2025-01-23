## LDM-DiT
该项目旨在使用LDM完成对全国区降水重建任务提供基线模型，该基线模型的输入有：卫星数据的7、8、11、13通道，size(1501, 1751)，实况数据的前4个通道，size(1501, 1751)，雷达数据的CR，size(1501, 1751)，时间分辨率为1h
## Setup
首先，下载并设置项目仓库：
```bash
git clone https://github.com/OpenEarthLab/CasCast.git
cd Dit
```

在 Python >= 3.10.0 的环境中安装 environment.yml：
```
conda env create -f environment
```

## Inferencing
在推理阶段，我们的输入是全国区(1501,1751)的数据，地址在./DataPath/Inf_allpath.txt给出，格式如目前txt文件所示，在加载测试数据集阶段，全国区(1501,1751)的测试数据会以步长128为overlap，划分成(256,256)的patch，划分好的数据地址与位置坐标信息会保存在./DataPath/Inf_patchpath.txt中，该过程是程序自动完成。
最终推理完成，我们会将所有的patch结果会以npy的形式保存至./SaveNpy/SavePatch文件夹
推理阶段所需的ckpt文件我们分别保存在：
diffusion的ckpt已经保存在DiT/ckpt/dif_checkpoint_latest.pth
autoEncode的ckpt已经保存在DiT/ckpt/ae_checkpoint_latest.pth
该信息已经在config文件与evalutation.py文件中设置完成，如果需要重新训练。请进行替换

```
bash ./scripts/eval_diffusion_infer.sh
```
通过上述命令获得patch的npy文件后，我们用下面的命令完成拼图

```
 python ./Puzzle.py
```
在上述程序中，p_path是指推理阶段patch文件的路径，即./DataPath/Inf_patchpath.txt，timetamp是指全国区数据的路径，即./DataPath/Inf_allpath.txt

运行完上述程序，我们会得到(1501,1751)的全国区拼图npy文件，保存在./SaveNpy/SaveAll。随后，我们将他进行插值至(6001,7001)，保存成nc文件，存储路径为./SaveNpy/Savenc
## Training

### step1. 训练一个Autoencoder
```
bash ./scripts/train_autoencoder.sh
```

### step2. 将用于训练与测试的降水图片由(256,256)，通过训练的Autoencoder，压缩到(32,32)的隐空间
```
bash ./scripts/compress_gt.sh
```

### step3. 训练Diffusion模型
```
bash ./scripts/train_diffusion_100M.sh
```
