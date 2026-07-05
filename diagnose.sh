
📄 diagnose.sh 完整代码

```bash
#!/bin/bash
# MORES 环境诊断脚本

echo "╔══════════════════════════════════════╗"
echo "║      MORES 环境诊断工具             ║"
echo "╚══════════════════════════════════════╝"
echo ""

# 检查 Docker
echo -n "🔍 Docker: "
if command -v docker &> /dev/null; then
    echo "✅ 已安装"
else
    echo "❌ 未安装"
    echo "   请访问 https://docker.com 下载安装"
fi

# 检查 Docker 是否运行
echo -n "🔍 Docker 服务: "
if docker info &> /dev/null; then
    echo "✅ 运行中"
else
    echo "❌ 未运行"
fi

# 检查端口 8000
echo -n "🔍 端口 8000: "
if netstat -an 2>/dev/null | grep -q ":8000.*LISTEN"; then
    echo "⚠️ 已被占用"
else
    echo "✅ 可用"
fi

# 检查 MORES 镜像
echo -n "🔍 MORES 镜像: "
if docker images --format "{{.Repository}}" 2>/dev/null | grep -q "mores-core"; then
    echo "✅ 存在"
else
    echo "❌ 不存在"
fi

# 测试 API
echo ""
echo "╔══════════════════════════════════════╗"
echo "║      测试 API 连接                   ║"
echo "╚══════════════════════════════════════╝"
if curl -s -X POST http://localhost:8000/decide \
    -H "Content-Type: application/json" \
    -d '{"input":"诊断测试"}' &> /dev/null; then
    echo "✅ API 服务: 正常响应"
else
    echo "❌ API 服务: 未响应"
fi

echo ""
echo "诊断完成。"
```

---

✅ 操作步骤

1. 将上面的代码复制
2. 粘贴到 diagnose.sh 文件内容框中
3. 提交信息：添加环境诊断脚本
4. 点击「提交」

---
