# EmbodiedHarness × SO-101

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

## 快速开始

需要 Python 3.11 或更新版本。仿真路径只使用 Python 标准库。

```bash
git clone <你的仓库地址>
cd harness
python3 -m scripts.run_sim_pick_place
python3 -m unittest discover -s tests -v
```

只有当最终状态验证和干净数据门控都通过时，仿真命令才会输出
`accepted simulation episode`。

## 安全边界

硬件不是仿真器的简单延伸。SO-101 工具要求已校准的从臂，并应用
[`configs/so101_safety.json`](configs/so101_safety.json) 中的约束。Safety Gate 会拒绝
未知关节、过期或越界状态、超过硬/软限位的目标、超过配置上限的单步动作，以及超过
时间预算的动作。

脚本采用保守策略：启用扭矩前先锁存实测位置，记录读回结果；除非一个通过验证的操作
显式要求暂时保持，否则默认通过断开扭矩来安全失败。请勿在未校准的机械臂上运行这些
工具。

## 尚未完成的部分

下一项硬件里程碑是末端视觉对齐和基座-桌面关系标定，然后是在目标物上方的闭环、仅接近式运动。碰撞几何、
三维工具位姿、力敏抓取和自主执行抓取仍必须经过额外验证；仓库不会将它们表述为已完成。

## 仓库卫生

仓库刻意排除标定图片、串口记录、生成的 episode 数据、虚拟环境和本地 PDF。这使公开
项目聚焦于可复现的源码、测试与设计决策。
