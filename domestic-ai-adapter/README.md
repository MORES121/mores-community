# 昇腾 HAL 抽象层适配（轻量版）

## 简介

本项目提供昇腾（Ascend）AI芯片的硬件抽象层（HAL）适配代码，支持 310B、910B4 等系列芯片。

总投入：1499元
- 硬件：Atlas 200I DK A2 开发板（1199元）
- 云上：华为云 ModelArts 910B4（300元）

一套代码，三款芯片兼容！

## 功能特性

- 张量接口（Tensor）
- 设备管理（Device）
- 执行器（Executor）
- 注册机制（Registry）
- 支持多后端扩展

## 编译运行

mkdir build && cd build
cmake ..
make
./test_hal

## 开源协议

Apache 2.0

---
墨睿思 MORES
