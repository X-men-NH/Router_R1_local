# 奖励设计

这份文档总结了 Router-R1 当前训练循环中实际使用的奖励设计，依据的实现文件是：

- `verl/utils/reward_score/qa_em.py`
- `verl/trainer/main_ppo.py`

这里描述的是训练时真实走到的代码路径，不是理想化设计。

## 当前这次 Run 的配置

最近这次 `linearplan-0410-01` 训练实际使用的是：

- `reward_metric = hybrid`
- `cost_coe = 0.0`

这意味着：答案质量使用 EM 加部分 F1 shaping 来打分；成本会被记录，但不会进入最终 reward。

## 总体公式

训练时的最终 reward 分两步计算：

1. 先计算一个由答案质量、格式、可选成本组成的基础分。
2. 再加上一个 decompose 辅助分。

代码里的最终训练 reward 是：

```text
reward_total = reward_base + decompose_aux_reward
```

对当前这次 run 来说，因为 `cost_coe = 0.0`，实际行为可以近似写成：

```text
if answer_reward == 0:
    reward_base = answer_reward + format_score
else:
    reward_base = answer_reward + effective_format_score

reward_total = reward_base + decompose_aux_reward
```

其中 `effective_format_score` 是“最终答案正确时可能被截断过的格式惩罚”。

## 答案分

系统会从最后一个 `<answer>...</answer>` 中抽取最终答案。

然后计算两个答案质量指标：

- `score_em`：归一化后是否和任一 gold answer 完全匹配
- `score_f1`：归一化后和 gold answers 的最佳 token-level F1

### Hybrid 模式

当前 run 使用的是 `reward_metric = hybrid`，因此：

```text
metric_score = score_em
answer_reward = score_em + f1_shaping_bonus(score_f1, score_em)
```

也就是说：日志里的 `metric` 显示的是 EM，但真正进入 reward 的答案分在 EM 之外还可能吃到一部分 F1 bonus。

### F1 Shaping Bonus

F1 shaping bonus 只会在以下情况下启用：

- `score_em < 1.0`
- `score_f1 > 0.4`

相关常量：

- `F1_SHAPING_THRESHOLD = 0.4`
- `F1_SHAPING_MAX_BONUS = 0.2`

行为可以概括为：

- 如果 F1 不超过 `0.4`，bonus 为 `0.0`
- 如果 EM 已经是 `1.0`，bonus 为 `0.0`
- 否则 bonus 随 F1 增长，但最大不超过 `0.2`

## 格式分

格式分由 `format_reward(completion)` 计算。

在当前实现里：

格式完全合法时：

```text
format_score = 0.0
```

格式错误现在分成三档：

```text
PUNISH_REWARD_MAX = -1.0
PUNISH_REWARD_MEDIUM = -0.5
PUNISH_REWARD_SMALL = -0.2
```

### 一个关键细节

代码里定义了三个惩罚常量：

- `PUNISH_REWARD_MAX`
- `PUNISH_REWARD_MEDIUM`
- `PUNISH_REWARD_SMALL`

它们现在已经被拆成三档：

- `PUNISH_REWARD_MAX = -1.0`
- `PUNISH_REWARD_MEDIUM = -0.5`
- `PUNISH_REWARD_SMALL = -0.2`

所以现在格式分不再是简单的二值设计，而是：

- `0.0`：格式完全合法
- `-0.2`：较轻微问题
- `-0.5`：中等级别问题
- `-1.0`：严重结构或协议错误

### 哪些情况会触发格式失败

格式检查会惩罚的情况包括：

- tag 缺失或不配对
- tag 嵌套
- 合法 tag 之外还有多余自由文本
- 缺少 `<think>` 或缺少最终 `<answer>`
- 出现多个 `<answer>`
- decompose 协议违规
- 没有 decomposition 却使用 `<subanswer>`
- 没有 decomposition 却使用 `[SubQk]`
- decompose 计划大小不合法
- search 格式不合法或 LLM 名称不合法

对于 decompose 轨迹，协议合法性还会额外通过 `analyze_decompose_protocol(...)` 检查。

当前三档大致对应：

- `PUNISH_REWARD_MAX = -1.0`
    严重结构或协议错误，例如 tag 不闭合、tag 嵌套、缺少最终 `<answer>`、decompose 协议非法、没有 decomposition 却使用 `<subanswer>` 或 `[SubQk]` 等。
- `PUNISH_REWARD_MEDIUM = -0.5`
    search 内容格式错误，例如仍然保留占位符、没有形成合法的 `ModelName:query`、query 为空等。
- `PUNISH_REWARD_SMALL = -0.2`
    search 的整体格式基本正确，但模型名不合法。

## 正确答案时的格式惩罚上限

对于完全正确的最终答案，系统有一个保护机制。

如果答案是对的，但格式分为负，那么这个格式惩罚会被截断：

```text
CORRECT_ANSWER_FORMAT_PENALTY_CAP = -0.2
effective_format_score = max(format_score, -0.2) when score_em == 1
```

这会防止“答案完全正确，却因为格式错误被整体打崩”。

### 例子

如果：

- `score_em = 1.0`
- `format_score = -1.0`
- 没有 decompose bonus

那么：

```text
effective_format_score = -0.2
reward_base = 1.0 - 0.2 = 0.8
```

这就是为什么日志里经常能看到：

```text
metric=1.0 format=-1.0 reward=0.8
```

## 成本分

API cost 会通过 `normalize_reward(token_price)` 转成一个类似 reward 的归一化值。

### 当前归一化方式

- 先对原始 token cost 做 `sqrt` 变换
- 使用长度为 `1000` 的滑动窗口
- 用分位数 `q_low = 0.05` 和 `q_high = 0.95` 做归一化
- 最后取反：`1.0 - scaled`

因此它的含义更接近“便宜程度奖励”：

- 越便宜，归一化后的 `api_cost` 越大
- 越贵，归一化后的 `api_cost` 越小

所以这里的 `api_cost` 不是原始成本，而更像“成本优势分”。

### 成本是否参与训练

只有在 `cost_coe > 0` 时，基础公式里才会混入成本：

```text
reward_base = (answer_reward + effective_format_score) * (1.0 - cost_coe) + api_cost * cost_coe
```

而当前这次 run：

```text
cost_coe = 0.0
```

所以成本虽然会被计算和记录，但不会改变最终 reward。

## Decompose 辅助分

在基础分算完之后，训练还会额外加一个 decomposition shaping term：

```text
reward_total = reward_base + decompose_aux_reward
```

这个辅助分只有在存在且仅存在一个 `<decompose>...</decompose>`，并且计划结构通过基本检查时才会生效。

### 常量

- `MAX_DECOMPOSE_BONUS = 0.18`
- `MIN_DECOMPOSE_BONUS = -0.05`
- `DECOMPOSE_STRUCTURE_BONUS = 0.01`
- `DECOMPOSE_SUBANSWER_BONUS = 0.02`
- `DECOMPOSE_ALL_DONE_BONUS = 0.03`
- `DECOMPOSE_UTILITY_MAX_BONUS = 0.05`
- `DECOMPOSE_EARLY_ANSWER_PENALTY = 0.03`
- `DECOMPOSE_UNKNOWN_ANSWER_PENALTY = 0.02`

### 逻辑

辅助分从 `0.0` 开始，然后按下面规则增减：

1. 只要 decompose 结构合法，先加 `+0.01`
2. 每解出一个唯一子问题，加 `+0.02`
3. 如果在第一次 final answer 之前已经完成所有子问题，再加 `+0.03`
4. 如果最终答案质量较好，再额外加最多 `+0.05`
5. 如果还有 TODO 时就提前给 final answer，减 `0.03`
6. 如果 final answer 是 unknown/none 一类答案，并且答案质量仍然为 0，再减 `0.02`
7. 最后整体截断到 `[-0.05, 0.18]`

### Utility Bonus

utility bonus 依赖于：

```text
utility_anchor = max(score_f1, score_em)
```

当 `utility_anchor > 0.5` 时，会按比例额外加分，上限是 `0.05`。

所以 decompose 不只是因为“结构上做了分解”而得分，也会因为它确实帮助生成了更有用的最终答案而得分。

## 一个特殊分支：答案分为 0

对于失败答案，代码走的是单独分支。

如果 `answer_reward == 0`，系统不会混入 cost，也不会使用“正确答案格式惩罚上限”，而是直接：

```text
reward_base = answer_reward + format_score
```

因此对于“答案错了 + 格式也错了”的情况，基础分通常就是：

```text
reward_base = 0 + format_score
```

也就是说，依据格式错误的严重程度，基础分可能是 `-1.0`、`-0.5` 或 `-0.2`。之后仍然会继续加上 decompose 辅助分。

## 日志里各字段代表什么

训练日志中：

- `metric`：在 `hybrid` 模式下表示 EM
- `format`：`format_reward` 的输出
- `reward`：加上 `decompose_aux_reward` 之后的最终 reward
- `api_cost`：归一化后的成本值
- `routes`：有效 `<search>` 调用的数量

这意味着日志里显示的 `reward` 永远是最终 reward，而不只是答案质量分。

## 示例

### 情况 1：答案正确，格式错误，没有 decompose bonus

```text
score_em = 1.0
score_f1 = 1.0
format_score = -1.0
effective_format_score = -0.2
decompose_aux_reward = 0.0

reward_total = 1.0 - 0.2 = 0.8
```

### 情况 2：答案错误，且是严重格式错误，没有 decompose bonus

```text
answer_reward = 0.0
format_score = -1.0
decompose_aux_reward = 0.0

reward_total = -1.0
```

### 情况 3：答案正确，并且有用的 decomposition 带来额外加分

```text
answer_reward ~= 1.0
effective_format_score = 0.0 or -0.2
decompose_aux_reward > 0

reward_total = answer_reward + effective_format_score + decompose_aux_reward
```

因此，有效的 decomposition 可以把最终 reward 推到高于“只有答案分”的水平。

## 当前实现中的几个重要注意点

### Prompt 和 Reward 还没有完全对齐

现在 prompt 已经被收紧为“只能拆成两个子问题”，但 reward 侧的协议代码在若干位置仍然接受长度为 2 或 3 的 decomposition plan。

这意味着 prompt 行为和 reward 校验目前还没有完全同步。

### 格式惩罚已经分级，但仍然偏粗糙

现在格式惩罚已经区分为 `-1.0 / -0.5 / -0.2` 三档，不再全部折叠成 `-1.0`。

不过它仍然相对粗糙，因为大量结构或协议错误仍然会统一落到 `-1.0`，更细粒度的协议学习信号仍然不够丰富。

### 成本目前没有参与优化

由于 `cost_coe = 0.0`，当前训练目标并不会鼓励更便宜的路由策略。成本现在只是可观察日志项，不是优化目标的一部分。

## 总结

对当前这次 run 来说，reward 实际上可以概括成：

```text
reward_total = answer_component + capped_or_uncapped_format_penalty + decompose_aux_reward
```

它的实际效果是：

- 完全答对仍然是主导项
- 部分 F1 只在 `hybrid` 模式下、且 EM 为 0 时才有帮助
- 格式失败现在有 `-1.0 / -0.5 / -0.2` 三档，但正确答案仍然会受到 `-0.2` cap 保护
- decomposition 可以提供适度正向 shaping，也可能带来小幅负向惩罚
- 路由成本目前只记录，不参与优化