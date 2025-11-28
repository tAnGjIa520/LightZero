# 性能监控设置指南

## 概述

本指南说明如何在 `train_alphazero.py` 中添加性能监控功能，用于监测 `collector.collect()` 的性能指标。

## 快速开始

### 1. 安装依赖

```bash
pip install psutil
# 如果需要详细GPU信息
pip install nvidia-ml-py3
```

### 2. 修改 train_alphazero.py

在文件顶部的导入部分添加：

```python
from lzero.utils.monitor import monitor_performance, print_system_info
```

在 `train_alphazero()` 函数开始处调用：

```python
def train_alphazero(...) -> 'Policy':
    # ... existing code ...

    # Add this line to print system info at startup
    print_system_info()

    # ... rest of the function ...
```

在主训练循环中修改 `collector.collect()` 调用（约125行）：

**原始代码：**
```python
new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
```

**修改后的代码：**
```python
with monitor_performance(f"collector.collect(iter={learner.train_iter})", logging.DEBUG):
    new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
```

### 3. 运行训练

```bash
python your_training_script.py
```

性能指标会自动打印到日志和stdout。

## 监控指标说明

### 输出示例

```
============================================================
Performance Report: collector.collect(iter=100)
============================================================
Duration: 5.23 seconds
CPU Usage: 45.3%
Memory (RSS): 8.45GB → 8.67GB (Δ +0.22GB)
GPU Memory: 2.34GB → 2.56GB (Δ +0.22GB)
============================================================
```

### 各指标含义

| 指标 | 单位 | 说明 |
|------|------|------|
| Duration | 秒 | 该操作总耗时 |
| CPU Usage | % | CPU平均使用率 |
| Memory (RSS) | GB | 进程内存（初始→结束→变化） |
| GPU Memory | GB | GPU显存（初始→结束→变化） |

## 高级用法

### 1. 监控多个操作

```python
with monitor_performance("data_collection"):
    new_data = collector.collect(...)

with monitor_performance("policy_update"):
    policy.learn_mode.learn(...)

with monitor_performance("evaluation"):
    stop, returns = evaluator.eval(...)
```

### 2. 保存指标到文件

```python
import json
from lzero.utils.monitor import ResourceMonitor

metrics_history = []

for iteration in range(num_iterations):
    monitor = ResourceMonitor(f"Iteration {iteration}")
    monitor.start()

    new_data = collector.collect(...)

    monitor.stop()
    metrics = monitor.get_metrics()
    metrics['iteration'] = iteration
    metrics_history.append(metrics)

# Save to JSON
with open('collection_metrics.json', 'w') as f:
    json.dump(metrics_history, f, indent=2)
```

### 3. 集成到TensorBoard

```python
from lzero.utils.monitor import monitor_performance

with monitor_performance("collector.collect") as monitor:
    new_data = collector.collect(...)

metrics = monitor.get_metrics()

# Write to tensorboard
tb_logger.add_scalar('collection/duration', metrics['duration_sec'], learner.train_iter)
tb_logger.add_scalar('collection/cpu_percent', metrics['cpu_percent'], learner.train_iter)
if 'gpu_memory_delta_gb' in metrics:
    tb_logger.add_scalar('collection/gpu_memory_delta_gb',
                        metrics['gpu_memory_delta_gb'],
                        learner.train_iter)
```

### 4. 集成到WandB

```python
import wandb
from lzero.utils.monitor import monitor_performance

with monitor_performance("collector.collect") as monitor:
    new_data = collector.collect(...)

metrics = monitor.get_metrics()
wandb.log({
    'collection/duration_sec': metrics['duration_sec'],
    'collection/cpu_percent': metrics['cpu_percent'],
    'collection/memory_delta_gb': metrics.get('memory_rss_delta_gb', 0),
    'collection/gpu_memory_delta_gb': metrics.get('gpu_memory_delta_gb', 0),
})
```

## 文件位置

| 文件 | 说明 |
|------|------|
| `lzero/utils/monitor.py` | 监控工具核心库 |
| `lzero/utils/monitor_example.py` | 使用示例和演示 |
| `MONITOR_SETUP.md` | 本文档 |

## 测试监控工具

```bash
# 运行示例代码
python -m lzero.utils.monitor_example

# 或者
cd /path/to/LightZero
python lzero/utils/monitor_example.py
```

预期输出：
```
============================================================
System Information
============================================================
Physical CPUs: 8
Logical CPUs: 16
...
GPUs Found: 1
  GPU 0: NVIDIA RTX A100 (40.00GB)
============================================================

============================================================
Example 1: Basic Monitoring
============================================================
...
```

## 故障排除

### 问题1：找不到psutil模块

**解决方案**：
```bash
pip install psutil
```

### 问题2：GPU信息无法获取

**原因**：PyTorch未安装或CUDA不可用

**解决方案**：
```bash
# 如果没有GPU，监控会跳过GPU部分，但不会报错
# 确保PyTorch已安装
pip install torch
```

### 问题3：权限不足无法监控GPU

**原因**：`nvidia-smi` 需要权限

**解决方案**：
```bash
# 作为普通用户运行，GPU监控仍然有效
# 如需详细GPU温度等信息，可能需要管理员权限
```

## 性能注意事项

### 监控本身的开销

监控工具的开销很小：
- CPU使用率采样：< 10ms
- 内存读取：< 5ms
- GPU内存查询（如果可用）：< 50ms

**总开销**：< 100ms per monitoring call

### 推荐做法

1. **生产环境**：只监控关键操作（数据收集、策略更新）
2. **调试环境**：可以监控更多细粒度的操作
3. **日志级别**：
   - `logging.DEBUG`：频繁操作
   - `logging.INFO`：主要操作
   - `logging.WARNING`：异常情况

## 自定义监控

### 扩展 ResourceMonitor 类

```python
from lzero.utils.monitor import ResourceMonitor

class CustomMonitor(ResourceMonitor):
    def get_metrics(self):
        metrics = super().get_metrics()
        # Add custom metrics
        metrics['custom_field'] = "custom_value"
        return metrics
```

### 添加自定义指标

```python
from lzero.utils.monitor import ResourceMonitor

monitor = ResourceMonitor("my_operation")
monitor.start()

# ... do work ...

monitor.stop()

# Add custom data
metrics = monitor.get_metrics()
metrics['custom_metric'] = 42
metrics['batch_count'] = 1000

# Log or save custom metrics
```

## 参考资源

- [psutil 文档](https://psutil.readthedocs.io/)
- [PyTorch CUDA 内存管理](https://pytorch.org/docs/stable/notes/cuda.html)
- [TensorBoard 集成](https://tensorboardX.readthedocs.io/)
- [WandB 集成](https://docs.wandb.ai/)

## 联系支持

如有问题，请参考：
- 代码注释和文档字符串
- `lzero/utils/monitor_example.py` 中的示例
- LightZero GitHub issues
