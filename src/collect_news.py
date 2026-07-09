"""
采集粤语新闻和社交媒体文本
"""
import json
import requests
import re

def collect_cantonese_news():
    print('📥 采集粤语新闻文本...')
    
    # 预置粤语新闻语料
    news_texts = [
        {'text': '今日香港天氣炎熱，最高氣溫達33度', 'source': 'news'},
        {'text': '政府宣布新一輪防疫措施', 'source': 'news'},
        {'text': '本港股市反覆上升，成交額增加', 'source': 'news'},
        {'text': '教育局公布新學年安排', 'source': 'news'},
        {'text': '警方呼籲市民注意網絡安全', 'source': 'news'},
        {'text': '運輸署提醒駕駛人士注意交通安全', 'source': 'news'},
        {'text': '醫管局加強醫院防疫措施', 'source': 'news'},
        {'text': '天文台預測未來數日持續有雨', 'source': 'news'},
        {'text': '多個商場推出夏日優惠活動', 'source': 'news'},
        {'text': '市民關注通脹對生活嘅影響', 'source': 'news'},
        {'text': '專家建議市民多做運動保持健康', 'source': 'news'},
        {'text': '學校陸續舉辦畢業典禮', 'source': 'news'},
        {'text': '政府推動智慧城市發展計劃', 'source': 'news'},
        {'text': '市民可透過手機應用程式查詢服務', 'source': 'news'},
        {'text': '本港旅遊業逐步復蘇', 'source': 'news'},
        {'text': '飲食業界推出多項優惠吸引顧客', 'source': 'news'},
        {'text': '警方加強巡邏確保節日安全', 'source': 'news'},
        {'text': '衛生署呼籲市民接種疫苗', 'source': 'news'},
        {'text': '教育局推動國情教育', 'source': 'news'},
        {'text': '多個團體舉辦文化活動', 'source': 'news'},
    ]
    
    with open('data/corpus_news.json', 'w', encoding='utf-8') as f:
        json.dump(news_texts, f, ensure_ascii=False, indent=2)
    
    print(f'✅ 采集完成！{len(news_texts)} 条新闻文本')

if __name__ == '__main__':
    collect_cantonese_news()
