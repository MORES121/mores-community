"""
修复 decision_engine.py 决策解释
"""

import re

with open('code/decision_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 确保决策解释正确传递稳定性
old_decision = 'explanation = self._explain(action, trajectory)'
new_decision = 'explanation = self._explain(action, trajectory, eval_result)'
content = content.replace(old_decision, new_decision)

with open('code/decision_engine.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ decision_engine.py 解释逻辑已修复')
