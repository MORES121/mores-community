"""
修复 counterfactual.py 评分逻辑 - 变量名错误
"""

import re

with open('code/counterfactual.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 修复 evaluate 方法中的变量引用错误
old_success_rate = 'success_rate = 1.0 if eval_result.get(\"success\", False) else 0.0'
new_success_rate = 'success_rate = 1.0 if sim_result.get(\"success\", False) else 0.0'

content = content.replace(old_success_rate, new_success_rate)

with open('code/counterfactual.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ counterfactual.py 变量名已修复')
