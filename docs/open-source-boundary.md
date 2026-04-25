# 昇腾 HAL 适配 - 开源边界说明

## 开源部分（Apache 2.0）

| 文件 | 说明 |
|------|------|
| include/hal/tensor.h | 张量接口定义 |
| include/hal/device.h | 设备接口定义 |
| include/hal/executor.h | 执行器接口定义 |
| include/hal/registry.h | 注册机制定义 |
| src/hal/registry.cpp | 注册机制实现 |
| src/test/test_hal.cpp | 编译测试程序 |
| CMakeLists.txt | 基础构建配置 |
| README.md | 项目说明 |
| LICENSE | Apache 2.0 许可证 |

这些文件可以在无昇腾硬件环境下编译验证。

## 闭源/商业部分（非公开）

| 文件 | 说明 |
|------|------|
| src/backend/shengteng/shengteng_device.cpp | 昇腾设备实现 |
| src/backend/shengteng/shengteng_executor.cpp | 昇腾执行器实现 |
| src/test/test_infer.cpp | 昇腾推理测试程序 |
| 性能优化版代码 | 内存池、多流并行 |
| 批量推理优化 | inferBatch 完整实现 |
| 寒武纪思元后端 | 第二芯片适配 |

## 用户验证路径

### 仅验证 HAL 接口（无需硬件）
git clone https://gitee.com/moshi-lab/mores-community.git
cd mores-community/domestic-ai-adapter
mkdir build && cd build
cmake .. && make && ./test_hal

### 完整昇腾推理（需要硬件或云环境）
- 联系我们获取商业授权版本
- 或自行实现 ShengTengDevice 和 ShengTengExecutor

## 商业咨询

**公司：东莞市轩钰希智能科技有限公司**

如需昇腾完整适配方案、性能优化版、寒武纪后端，请联系：klidwinac@yeah.net
