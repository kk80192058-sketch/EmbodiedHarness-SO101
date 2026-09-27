# EmbodiedHarness × SO-101

[![CI](https://github.com/kk80192058-sketch/EmbodiedHarness-SO101/actions/workflows/ci.yml/badge.svg)](https://github.com/kk80192058-sketch/EmbodiedHarness-SO101/actions/workflows/ci.yml)

一个带有**证据门控（evidence-gated）**机制、与具体模型无关的运行时，用于探索
**零示教桌面操作**。项目将高层任务推理与真实硬件执行解耦，使每一条真实动作都受到
边界约束、被完整记录，并可由独立证据复核。

> 当前状态：仿真 Agent 运行时与 SO-101 的安全 bring-up 工具已经验证可用。
> 项目刻意**不宣称**已经实现真实环境中的全自主抓取与放置。

## 为什么做这个项目

很多桌面机械臂演示直接把视觉结果发送到电机。本项目关注的是让这段交接过程可审计、
可复现的基础能力：

- 有类型约束的观测、技能、结果与证据契约；
- 在尚无示教数据时也能工作的确定性几何技能；
- 有上界的恢复机制，而不是无限重试；
- 只追加写入的 episode 轨迹与最终状态验证；
- 数据门控：把失败或经历恢复的轨迹排除在自我改进训练数据之外；
- 面向 SO-101 从臂的硬件 Safety Gate：在电机收到目标前拦截不安全指令。

## 架构

```text
任务请求
  -> 确定性技能规划
  -> 仿真器或未来的机器人适配器
  -> 只追加写入的 JSONL 轨迹
  -> 最终状态证据验证器
  -> 干净数据准入门
```

任务 Agent 与数据准入的不变量见
[`docs/agent_runtime.md`](docs/agent_runtime.md)。

## 已实现内容

| 层次 | 实现 | 验证方式 |
| --- | --- | --- |
| 类型化契约 | `harness/core.py` | 单元测试 |
| 确定性桌面仿真 | `harness/sim.py` | 可回放的 pick/place episode |
| 几何技能 | `harness/skills.py` | 前置条件与最终状态检查 |
| Agent 与恢复机制 | `harness/agent.py` | 有界重试与中止测试 |
| 数据质量门控 | `harness/data_gate.py` | 干净轨迹与恢复轨迹的区分测试 |
| 电机 Safety Gate | `harness/safety.py` | 硬/软限位与超时测试 |
| SO-101 bring-up 工具 | `scripts/safe_so101_step_test.py`、`scripts/so101_hold_watchdog_test.py` | 仅在完成实体校准后使用 |
| 视觉工作区工具 | `scripts/calibrate_tabletop_from_a4.py`、`scripts/detect_visual_proxies.py`、`scripts/track_gripper_star_template.py` | 本地、贴近硬件的标定工具；模板工具只验证二维视觉代理的可重复性 |
| 基座对齐工具 | `scripts/calibrate_robot_base_from_contacts.py` | 由至少三个经确认的夹爪中心桌面接触点拟合刚体二维变换；残差不合格时拒绝使用 |
| 双相机 SO-101 会话 | `harness/so101_session.py`、`scripts/so101_session.py` | 持久串口连接下同步保存腕部/全局图、完整寄存器和时间戳；默认只读 |
| SmolVLA 影子策略 | `harness/smolvla_policy.py`、`scripts/smolvla_shadow.py` | 固定修订、本地权重校验、双相机离线推理、单位转换和不可执行性报告；绝不直接写电机 |
| 实体会话离线分析 | `harness/session_analysis.py`、`scripts/analyze_so101_sessions.py` | 按事件顺序严格配对每次写入、结果和相邻图像；错配或不完整区间不能进入局部响应估计 |
| 策略就绪度报告 | `harness/so101_readiness.py`、`scripts/assess_so101_readiness.py` | 汇总状态、双相机新鲜度、目标唯一性与影子策略兼容性；只输出证据，永不授权电机动作 |
| 接触样本证据审计 | `harness/contact_evidence.py`、`scripts/audit_so101_contact_evidence.py` | 校验人工接触样本及其相机证据，并输出输入内容哈希；不把完整性审计误作实体标定成功 |
| 接触姿态覆盖诊断 | `harness/contact_coverage.py`、`scripts/assess_so101_contact_coverage.py` | 将保存的当前关节状态与审计过的接触样本逐关节比较；包络外的姿态不得被表述为已观测接触姿态 |

## 快速开始

需要 Python 3.11 或更新版本。仿真路径只使用 Python 标准库。

```bash
git clone <你的仓库地址>
cd harness
python3 -m pip install -e ".[test]"
python3 -m scripts.run_sim_pick_place
python3 -m unittest discover -s tests -v
```

只有当最终状态验证和干净数据门控都通过时，仿真命令才会输出
`accepted simulation episode`。

SmolVLA 影子推理是一个单独的可选环境：`python3 -m pip install -e ".[so101-policy]"`。
它依赖本地模型权重、已校准硬件和双相机保存证据；安装它不会启用或授权实体动作。

## 开发与贡献

GitHub Actions 会在 Python 3.11 和 3.12 上运行独立的仿真、标定、日志配对与安全契约测试。
提交前可运行：

```bash
python3 -m unittest discover -s tests -v
git diff --check
```

贡献规范见 [`CONTRIBUTING.md`](CONTRIBUTING.md)，安全问题请遵循
[`SECURITY.md`](SECURITY.md) 的私密披露流程。仓库尚未选择开源许可证；在复用或分发前，
请先取得维护者的明确许可。

## 安全边界

硬件不是仿真器的简单延伸。SO-101 工具要求已校准的从臂，并应用
[`configs/so101_safety.json`](configs/so101_safety.json) 中的约束。Safety Gate 会拒绝
未知关节、过期或越界状态、超过硬/软限位的目标、超过配置上限的单步动作，以及超过
时间预算的动作。

脚本采用保守策略：启用扭矩前先锁存实测位置，记录读回结果；除非一个通过验证的操作
显式要求暂时保持，否则默认通过断开扭矩来安全失败。请勿在未校准的机械臂上运行这些
工具。

## 尚未完成的部分

2026-09-27 实体接管状态、动作证据、A4 映射问题及下一步验收要求见
[`docs/so101_resume_20260927.md`](docs/so101_resume_20260927.md)。受限动作入口
`scripts/bounded_so101_hold_step.py` 缺省只读，并显式区分到位、滞住和保护停止；
它尚不是完整的实体抓取后端。

同日已经验证本地固定修订的 SmolVLA 可以对实际双相机证据完成影子推理，但它的首步
夹爪提议超出已校准软范围，且现场腕部/夹爪状态偏离训练统计。策略输出因而只能作为
诊断记录，不能自动升级为实体命令；完整结论与复跑命令见
[`docs/so101_policy_integration.md`](docs/so101_policy_integration.md)。

下一项硬件里程碑是末端视觉对齐和基座-桌面关系标定，然后是在目标物上方的闭环、仅接近式运动。碰撞几何、
三维工具位姿、力敏抓取和自主执行抓取仍必须经过额外验证；仓库不会将它们表述为已完成。

新的人工接触采样必须由操作者将夹爪中心放在已知 A4 点后，以仓库根目录为工作目录，显式使用
`python -m scripts.capture_so101_contact_sample --confirm-jaw-centre`。采样会绑定当时的
电机标定 JSON 与 URDF 的 SHA-256；审计器会拒绝声称采用这一 v2 采样格式、但来源文件
缺失或内容已改变的样本。此确认和哈希只增强可追溯性，仍不等同于软件已验证实际接触。

## 仓库卫生

仓库刻意排除标定图片、串口记录、生成的 episode 数据、虚拟环境和本地 PDF。这使公开
项目聚焦于可复现的源码、测试与设计决策。
