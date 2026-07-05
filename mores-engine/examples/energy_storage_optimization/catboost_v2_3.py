"""
墨睿思MORES - CatBoost v2.3
基于v2.0最佳配置，微调特征
"""
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from pathlib import Path
from datetime import datetime

DATA_DIR = Path('C:/energy_optim/data')

class MORESEngineV23:
    def __init__(self):
        self.model = None
        self.decision_log = []
        self.random_seed = 42
        
    def load_data(self):
        print("[1/5] 加载数据...")
        df_feat = pd.read_csv(DATA_DIR / 'train/mengxi_boundary_anon_filtered.csv')
        df_label = pd.read_csv(DATA_DIR / 'train/mengxi_node_price_selected.csv')
        df = pd.merge(df_feat, df_label, on='times', how='inner')
        df['times'] = pd.to_datetime(df['times'])
        df = df.sort_values('times')
        return df
    
    def feature_engineering(self, df):
        print("[2/5] 特征工程 v2.3...")
        # 时间特征
        df['hour'] = df['times'].dt.hour
        df['dayofweek'] = df['times'].dt.dayofweek
        df['month'] = df['times'].dt.month
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # 季节特征
        df['season'] = df['month'].apply(lambda x: 0 if x in [12,1,2] else 1 if x in [3,4,5] else 2 if x in [6,7,8] else 3)
        
        # 供需平衡（核心）
        df['供需平衡'] = df['风光总加预测值'] - df['系统负荷预测值']
        
        # 供需平衡变化率（v2.0最重要特征）
        df['供需平衡_diff'] = df['供需平衡'].diff(1)
        df['供需平衡_diff_rate'] = df['供需平衡_diff'] / (abs(df['供需平衡']).rolling(24).mean() + 0.01)
        
        # 简化特征集（保留Top重要特征）
        self.feature_cols = [
            'hour_sin', 'hour_cos', 'dayofweek', 'month', 'season',
            '供需平衡', '供需平衡_diff_rate',
            '风电预测值', '光伏预测值', '系统负荷预测值'
        ]
        
        print(f"   使用 {len(self.feature_cols)} 个特征（精简版）")
        return df
    
    def train(self, df):
        print("[3/5] 训练CatBoost趋势预测模型 v2.3...")
        df['price_change'] = (df['A'].shift(-1) - df['A']) > 0
        df['price_change'] = df['price_change'].astype(int)
        df = df.dropna()
        
        X = df[self.feature_cols].values
        y = df['price_change'].values
        
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        # v2.0最佳参数
        self.model = CatBoostClassifier(
            iterations=400,
            learning_rate=0.025,
            depth=7,
            l2_leaf_reg=2,
            random_seed=self.random_seed,
            verbose=False
        )
        self.model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        
        importance = dict(zip(self.feature_cols, self.model.feature_importances_))
        sorted_feat = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        print(f"   Top5特征:")
        for i in range(5):
            print(f"      {i+1}. {sorted_feat[i][0]}: {sorted_feat[i][1]:.1f}")
        return df
    
    def predict_trend(self, df_test):
        X_test = df_test[self.feature_cols].values
        return self.model.predict_proba(X_test)[:, 1]
    
    def decide_strategy(self, trends, day):
        actions = np.zeros(96)
        duration = 8
        power = 1000
        max_start = 96 - duration
        
        charge_sums = [trends[i:i+duration].sum() for i in range(max_start + 1)]
        best_charge = int(np.argmin(charge_sums))
        start_min = best_charge + duration + 8
        
        if start_min <= max_start:
            discharge_sums = [trends[i:i+duration].sum() for i in range(start_min, max_start + 1)]
            if discharge_sums:
                best_discharge = int(start_min + np.argmax(discharge_sums))
                actions[best_charge:best_charge+duration] = -power
                actions[best_discharge:best_discharge+duration] = power
                
                self.decision_log.append({
                    'day': int(day),
                    'charge_start': int(best_charge),
                    'discharge_start': int(best_discharge),
                    'timestamp': datetime.now().isoformat()
                })
        return actions
    
    def run(self):
        print("=" * 50)
        print("墨睿思MORES - CatBoost v2.3")
        print("优化方向: 精简特征集 | 保留核心特征")
        print("=" * 50)
        
        df_train = self.load_data()
        df_train = self.feature_engineering(df_train)
        self.train(df_train)
        
        print("[4/5] 预测测试集...")
        df_test = pd.read_csv(DATA_DIR / 'test/test_in_feature_ori.csv')
        df_test['times'] = pd.to_datetime(df_test['times'])
        df_test = self.feature_engineering(df_test)
        df_test = df_test.fillna(0)
        
        trends = self.predict_trend(df_test)
        print(f"   趋势范围: {trends.min():.4f} ~ {trends.max():.4f}")
        
        print("[5/5] 生成充放电策略...")
        all_actions = []
        num_days = len(df_test) // 96
        for day in range(num_days):
            day_trends = trends[day*96:(day+1)*96]
            actions = self.decide_strategy(day_trends, day+1)
            all_actions.extend(actions)
        
        output_df = pd.DataFrame({
            'times': df_test['times'][:len(all_actions)],
            '实时价格': 0,
            'power': np.array(all_actions).astype(int)
        })
        output_df.to_csv(DATA_DIR.parent / 'submissions' / 'output.csv', index=False)
        
        trade_days = len(self.decision_log)
        print(f"\n✅ 运行完成！")
        print(f"   操作天数: {trade_days}/59")
        print(f"   Power分布: {output_df['power'].value_counts().to_dict()}")
        print("=" * 50)
        print("墨睿思MORES v2.3: 可控 | 可解释 | 可追溯 | 全自动")
        return output_df

if __name__ == '__main__':
    engine = MORESEngineV23()
    engine.run()