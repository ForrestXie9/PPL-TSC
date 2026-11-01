# 教师实现核对与迁移协议

## 代码来源与版本

参考来源为本地存档的 [firepd/PI-Light](https://github.com/firepd/PI-Light) 源码。
保留未修改的参考文件在 `reference/pi_light/`，不是声称从各教师论文官方仓库重新取得。
当前运行代码包含既有修复，与参考文件的逐行差异见 `teacher_*.patch`，文件哈希见
`TEACHER_IMPLEMENTATION_AUDIT.json`。不能把当前运行代码称作逐字未经修改的上游原版。

本次未改变 PPL-TSC 搜索、标定、评分、提示词；也未把 FRAP/EcoLight 改成共享训练。
原研究目录中的 Shared 实验不受本次打包修订影响，旧实验结果不能自动当作原版教师结果。

## 各教师

| 教师 | 原参数组织 | 当前运行代码与参考的关系 | 跨路网 |
|---|---|---|---|
| FRAP | 每路口独立网络 | 保留原组织；既有设备适配，详见 patch | 源 agent 0 参数复制；目标重新构建相位/交通流向映射 |
| EcoLight | 每路口独立网络；2 输入、2 输出的保持/切换网络 | 保留原组织；既有设备适配及按实际观测相位解码动作修复 | 源 agent 0 参数复制；目标重新构造绿/红需求和相位映射 |
| MPLight | 原本跨路口共享层和 optimizer | mplight.py 与参考文件逐字一致；继承的 FRAP 有设备适配 | 源 agent 0 参数复制；目标仍按原实现共享 |
| CoLight | 原本共享编码、注意力、输出层 | 有既有异构相位处理、有效动作裁剪、观测 padding/truncation、梯度裁剪等改动，详见 patch | 输入/输出维度受路网约束，仅兼容时加载；不能把“共享”当作任意路网可迁移 |
| TinyLight | 每路口独立结构，输入/输出随车道和相位数变化 | 有既有设备、架构参数梯度修复，详见 patch；当前是未剪枝训练版本 | 原 baseline-transfer 脚本没有确立其协议，本包只提供同路网 |

CoLight 的观测 padding/truncation 是既有运行实现行为，本次没有新增。它可能丢失观测，
即便权重形状兼容也不意味着语义正确或效果可靠。对原论文逐字复现，应使用参考文件核对
这些既有改动，而不能隐去它们。迁移加载器本身不截断/补齐参数，不忽略不匹配的键。

## 上游迁移脚本实际做什么

`015_baseline_transfer.py` 在源路网训练 64 个 episode，随后：

1. 创建目标环境和目标路口 agent，使用目标路网的相位映射。
2. 对每个目标路口复制 `env.n_agent[0].network_local` 的学习参数。
3. 在目标环境执行推理，不训练、不存经验。

这对多路口独立 FRAP/EcoLight 的含义是**只迁移源路口 0 的网络**，不是迁移所有源路口，
也不是共享训练。源路口 0 在测试前固定，不按目标 ATT 挑选。

原脚本逐参数 zip/copy，本包用 strict state_dict 检查，先核对全部键和形状再加载。
目标的拓扑映射不是从源路口硬拷贝。源模型与学生必须来自同一个明确的源训练任务；
若学生利用所有源路口轨迹，而教师迁移只取 agent 0，报告中也必须注明这一不对称。

## 使用

`evaluate_teacher.py --protocol same_network`：同路网，保留各路口本来的权重。
必须通过路网哈希检查，适用于原场景和同路网跨流量。

`evaluate_teacher.py --protocol source_agent0`：跨路网，使用上述预先固定的源 agent 0。
输入 checkpoint 必须与训练 audit 哈希一致。拒绝额外 Shared 训练产物；兼容失败时记录失败
而不静默修改模型。CoLight 允许尝试但不承诺兼容；TinyLight 显式拒绝此未经确立的协议。

默认训练入口仍是本项目已修复的 128-episode、训练 ATT 最优 checkpoint 协议；它与上游
64-episode 迁移脚本不是完全相同的训练设置。可用 `--episodes 64 --selection last` 接近该
训练长度及最终权重选择，但这并不能消除其他实现/环境差异。

## 本次删除的内容

干净包删除新增的 `FRAP_Shared_Teacher_Retrain`、`EcoLight_Shared_Teacher_Retrain`
及其加载器/采集选项。MPLight、CoLight 原生的共享层不删除。

## 本次实际验证

五类教师同路网跨流量均完成；普通 FRAP/EcoLight/MPLight 的杭州1→杭州2、杭州1→Atlanta 都完成。CoLight 杭州1→杭州2可运行；杭州1→Atlanta因输入维度22→15、输出相位6→5被严格拒绝。
这些是2-episode临时模型的接口测试，不是性能结论。完整记录见 `TEACHER_TRANSFER_VALIDATION.json`。
