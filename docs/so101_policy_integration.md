# SO-101 预训练策略接入诊断（2026-09-27）

## 实测现状

- `TaskAgent` / `GeometricSkillBackend` 当前接 `TabletopSimAdapter`；尚无实机 pick/place 技能后端。
- 当前真实执行入口是单关节 bring-up 脚本，每次重新连接相机与串口。11:59–12:44 的 34 段动作，调用起点间隔中位数 66 秒，事件日志覆盖时段中位数 1.682 秒。二者不构成完整延迟分解，但足以说明主要等待并非电机运动时间。
- 20260927T144425 已只读拍摄两只 USB 相机：OpenCV 0 为腕部近景，1 为全局桌面。之前脚本只调用 1，遗漏了已有腕部视角。设备编号可能在重连后变化，运行前须核对画面与设备身份。
- 本机 Apple Silicon、16 GiB 内存，PyTorch 2.11.0 MPS、LeRobot 0.6.1 与 transformers 5.5.4 可用；固定权重已通过本地实际加载，真实观测的影子推理延迟见下方结果。
- 原校准继续有效保存；新策略不能默认调用重新归零、关扭矩或覆盖校准流程。

### 已完成的真实证据影子验证（17:25）

固定模型权重已完整下载并且 SHA-256 与保存的 Hub 元数据匹配。对
`artifacts/bringup/sessions/1790502695828083000/0003_observation/observation.json`
及其两张保存图像，`scripts/smolvla_shadow.py` 使用 MPS 实际加载模型并运行两次。
首次加载后的推理耗时约 0.889 秒，第二次约 0.495 秒。两次首步均无法转换成已校准
原始编码器目标：夹爪提议约为 1802 / 1807，低于 `range_min=1849`。

该观察的处理后状态中 wrist_flex 约为 −3.58、gripper 约为 −5.83，超过训练统计的
3σ 门。`assess_proposal` 同时保留这些偏离、转换错误和各关节步进验证结果；它固定
返回 `hardware_execution_permitted: false`，且影子模块没有总线或相机访问路径。
这验证了“可加载、可推理、可拒绝”的集成阶段，**没有**验证策略适配、动作执行或抓取。

### 统一运行时就绪度（17:33）

`harness/so101_readiness.py` 把一份保存观测和一份保存的影子策略回执汇总为只读
就绪度报告。它单独检查关节遥测、两路图像的采集年龄、红块唯一可见性、影子回执确实
零写入，以及策略兼容性。无论检查结果如何，`motion_authorized` 恒为 `false`：该模块不是
执行器，也不能成为绕过现有单关节监督器的通道。

对最新证据生成的报告在
`artifacts/policy_readiness/1790503019978004000.json`。结果为
`observation_healthy: true`，说明电机状态与双相机观测完成了各自的健康门；但
`policy_proposal_compatible: false`，理由仍是训练分布偏离和夹爪目标低于校准下限。
报告还明确记录没有已验证的多关节实体执行器。这将“现场在线”与“可执行抓取”分离，
防止上层任务运行时把相机健康误解为动作许可。

可离线复跑：

```bash
.venv/bin/python -m scripts.assess_so101_readiness \
  --observation artifacts/bringup/sessions/1790502695828083000/0003_observation/observation.json \
  --prediction artifacts/policy_shadow/1790502703968809000/prediction_0.json
```

可重跑（不会打开串口或写电机）：

```bash
.venv/bin/python -m scripts.smolvla_shadow \
  --observation artifacts/bringup/sessions/1790502695828083000/0003_observation/observation.json \
  --model artifacts/models/so101_pick_place_smolvla/1c6badde8f3ddda0c546c4988a92c338ae74f22a \
  --vlm artifacts/models/smolvlm_processor/7b375e1b73b11138ff12fe22c8f2822d8fe03467 \
  --body-units degrees --task pick_and_place --runs 2
```

双相机证据：`artifacts/bringup/camera_inventory/20260927T144425/`。
Hub 配置与修订号实查：`artifacts/policy_audit/20260927T144639/`。
此前一次 urllib TLS 证书链失败记录保留在 `20260927T144605/`；随后用系统 curl 正常验证 TLS 获取成功，没有关闭证书验证。

## 已发现的候选（未下载模型权重，未接管硬件）

| 模型 | 核查结果 | 用途与限制 |
| --- | --- | --- |
| `nota-gmbh/so101_pick_place_smolvla` | 修订 `1c6badde8f3ddda0c546c4988a92c338ae74f22a`；6 维 state/action；fixed 640×480、handy 1920×1080；50 步 chunk；含预处理、反归一化及统计文件 | 可优先做兼容性与离线推理验证；两视角结构与现有硬件可对应，但相机名称/分辨率匹配不证明视角和任务分布匹配 |
| `lerobot/smolvla_base` | 修订 `d9f33c94a60fb382c90dea2164c96845bd955e28`；3 个 camera 输入；6 维 state/action | 通用预训练起点，不能直接声称适配现场；空视角等处理须依据训练约定 |
| `ahmedsohail2003/smolvla-so101-pickplace-v2` | 修订 `f43f9f3a1c81b050f9747919c53019fcae2caec0`；模型卡报告仿真训练与评估 | 名称包含 SO101，仍不能将仿真成功率当作本机实体成功率 |

## 应实现的执行链

高层任务 Agent → 固定版本的策略与输入适配 → 双相机/关节状态 → SmolVLA 动作段
→ 本地动作执行器及连续监督 → 实体结果验证 → 完整 episode 数据。

模型只在任务、恢复或策略选择层做较慢决策。相机采集、动作段执行、状态监控应在本地持续运行。
控制频率需依据总线、相机和 MPS 推理实测确定，不能直接承诺 30 Hz。策略动作段也不能不经检查整段发给电机。

## 实现顺序与验收

1. **双相机与硬件适配器**：长连接、时间戳、角色验证；继续加载原校准。验收为持续采集两路图像与状态，无电机写入、无校准变化。
2. **策略发现与兼容性筛选**：检索后核查机器人种类、关节顺序、单位、动作表示、相机训练视角、训练任务、归一化统计及依赖；固定模型修订。下载权重前检查大小与可用空间。
3. **离线/影子推理**：加载完整模型和匹配的前后处理器，以真实观测计算但不发送动作；检查有限数值、单位转换、第一步偏差、动作段长度、延迟和内存。不允许把模型输出直接当原始编码数。
4. **受监督连续执行**：本地执行器处理速度、加速度、位置范围、跟随误差、状态陈旧、通信超时与中止；保留对话模型之外的停止机制。单关节 40 计数仍是现有明确授权，不能把它偷偷换成未验证多关节轨迹。
5. **实体任务后端和验收器**：将仿真布尔值替换为闭合、物体随动、提起/放下及最终位置等实测证据，接入 TaskAgent。
6. **经验与改进**：记录同步图像、关节状态、真实发送动作、策略原始输出、模型版本、任务和结果。成功轨迹、失败、纠正、恢复分别标注；失败轨迹不能直接作为成功模仿标签。增加数据集导出、训练/微调任务、固定评测集、版本晋升和回滚。只有同一评测分布上提高成功率，才能称为改进。

自动采集不会自动改变冻结权重。自主探索仍需要可验证奖励、受限空间与初始状态复位；这些组件现在尚未实现。
已有 motor calibration 应优先验证和复用；视觉/工具标定与策略输入适配属于不同问题，预训练权重不能替代机械硬限位校准。

## 参考

- SmolVLA 官方文档：https://huggingface.co/docs/lerobot/main/smolvla
- 模型与配置：https://huggingface.co/nota-gmbh/so101_pick_place_smolvla
- LeRobot 持续执行与 RTC：https://huggingface.co/docs/lerobot/main/inference
- GPT-6 Astra 机器人原作者实验：https://pantograph.com/journal/vlm-harness

Pantograph 的模型每轮获取多相机图像，使用参数化末端动作，由 harness 转成关节命令；这不是“裸模型直接发编码器目标”，也不证明演示使用 SmolVLA。
