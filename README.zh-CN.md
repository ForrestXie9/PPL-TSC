# PPL-TSC 干净源码包

本目录独立于研究目录，包含当前方法、五类 DRL 教师训练代码、轨迹采集、蒸馏和评测。
**不包含训练好的模型、教师轨迹、搜索得到的策略、历史实验、真实 API 密钥或虚拟环境。**

## 从零使用

1. Linux x86_64 安装 Python 3.10（含开发头文件和 venv）、C++ 编译工具和 bubblewrap。
2. 执行 `bash scripts/setup_venv.sh`，随后 `source .venv/bin/activate`。
3. 按 [README.md](README.md) 的完整命令训练教师；训练产物均写入 `outputs/`。
4. 使用 `collect_teacher.py` 从自己训练的 checkpoint 采集对齐轨迹。
5. 使用 `scripts/prepare_distillation.py` 生成当前方法配置；支持七种教师/学生接口。
6. 将 `.env.example` 复制为 `.env`，本地填入密钥。配置内可更换 endpoint、model。
7. 先 `run.py --dry-run --no-api-profile` 做离线预检，再执行蒸馏。蒸馏会产生第三方 API 费用。
8. 使用 `evaluate.py` 在原流量、同路网其他流量测试。

教师训练默认 128 个 episode，保留训练 ATT 最好的 checkpoint，可用 `--resume` 续训。
TinyLight 为本项目修复后的未剪枝 128-episode 版本，不能写成原始 TinyLight 的 64+128 剪枝协议。
EcoLight 提供三个独立入口：native 使用绿／红车辆数并比较 KEEP／SWITCH；旧版 naive 使用相位特征，只比较当前相位和循环下一相位；phase-selection 对全部相位评分。
配置工具为 naive 和 phase-selection 生成独立的物理相位轨迹投影，保留教师动作并记录原轨迹哈希。
FRAP、EcoLight 保留每路口独立训练；不提供后加的 Shared 训练变体。
MPLight、CoLight 保留原代码自带的参数共享。详见 [教师核对与迁移协议](docs/TEACHER_PROTOCOL.md)。

## 迁移规则

同路网保留源偏置与统计量。跨路网必须显式调用 `scripts/export_formula_transfer.py`，
生成移除源路口偏置的独立文件，然后评测；它保留公式及源统计量，**不是完整标定参数的迁移**。
不在目标流量上重新选择公式或调整参数。教师跨路网遵循 PI-Light 的源 agent 0 复制协议，使用 `evaluate_teacher.py --protocol source_agent0`。
CoLight 仅在维度及语义兼容时可用，TinyLight 原脚本未确立跨路网协议。

## 当前配置

`0.4.5-scale-reminder`；种群 5、5 代；fitness = ABF + 封顶 CP；偏置 ±0.5；
CMA-ES 种群 20、最多 20 迭代；保留原严格 fitness 父代替换规则。
新增当前相位奖励尺度提醒；不包含研究目录中的其他实验性替换规则。
验证器是有限输入检查，不是对任意输入的形式化安全证明，也不保证学生普遍优于教师。

## 发布前事项

代码包已整理为可独立运行目录，最新验证范围见 `docs/COMPREHENSIVE_REVIEW_VALIDATION.json`。
不把短程冒烟测试当成完整训练/付费蒸馏复现。
项目许可证、教师原代码及交通数据的来源授权仍须作者确认，第三方原始许可证已保留。
本候选版不包含个人环境配置。

## 模块复核

独立安装、五类教师在单路口/多路口短训、六种接口完整离线搜索、16次导出策略重放及42项测试已通过。
离线搜索用固定测试响应替代LLM，不代表真实付费生成质量或论文性能。
实际修复了/tmp下虚拟环境被隔离器挂载遮蔽的问题，安装结束增加隔离执行自检。
详见 `docs/MODULE_VALIDATION.json`。

### 轨迹来源核对

采集时默认读取模型旁边的 `audit.json`，核对训练完成状态、教师类别、模型哈希与源路网。
也可用 `--training-audit` 指定位置。只有缺少训练记录的旧模型才使用
`--allow-unaudited-checkpoint`；输出会标记为未核验，且此选项不能绕过已有记录的不匹配。
同路网可换流量采集，跨路网教师迁移请使用单独的显式迁移协议。
`--stop-after` 必须为正整数，续跑时不能早于已完成的训练轮次。

### 重试与续跑

实际启动时会保存 `run_identity.json`，并锁定输出目录。续用目录必须保持配置、
源码、教师轨迹、路网和流量文件不变；改变这些内容请使用新目录。
缺少身份记录的旧结果会保留，但不能自动复用。相同设置的失败任务可重试；
`--resume` 仍只支持顺序批量任务，仅跳过公式、偏置、配置和指标相互一致的完整结果。
批量任务中会写入同一路径的条目会在启动前拒绝，不同搜索种子的独立实验请分别指定输出目录。

### 失败结果与离线模式

评测结果会明确记录 `running`、`failed` 或 `complete`，统计时只使用 `complete`。
重跑开始后不会继续显示旧的成功结果；结果采用原子写入，同一结果文件禁止并发写入。
强制杀进程可能留下 `running`，不能当成成功。采集轨迹需使用新路径，失败时不会发布半份 gzip。
未知特征契约会报错；JSON 中的 API provider 优先于残留环境变量，离线模式禁止旧后端绕过缓存直接请求。

## 固定清单逐项验收

最新完整记录以 `docs/COMPREHENSIVE_REVIEW_VALIDATION.json` 为准；此前各报告是历史记录。
本轮固定源码执行 157 条检查：18 个测试文件独立执行、87 项回归测试、五类教师在
Hangzhou1/Atlanta 的 10 组两轮短训、10 次轨迹采集、七种学生接口完整离线蒸馏、
21 次学生评测、教师迁移与两类续跑检查，均符合预期。
其中不支持的教师跨路网情况是明确拒绝，不能当作成功迁移。
另有 128 轮教师训练、真实 LLM 蒸馏和随机案例验收；本包不附训练产物或私有日志。

```bash
python scripts/audit_release.py --output-dir ../PPL_TSC_Audit_Run
```

输出目录必须不存在且位于源码目录之外。脚本按顺序执行，逐步写入 `audit_report.json`
和日志；失败立即记录，后续未执行项保留 `not_run`，不会写成全部通过。
检查期间源码不得变动。教师短训为 2 episodes；蒸馏保留 5×5 和 CMA-ES 20×20 上限，
但将 LLM 响应替换为确定的测试代码并禁止网络请求。实际调用验证器、标定、仿真、
评分、选择、导出与续跑。它验证运行链路，不验证真实 LLM 质量、完整训练收敛或 GPU。
