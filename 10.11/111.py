import pandas as pd
import numpy as np
import re
import json
import os
from collections import Counter

# ========== 1. 读取数据 ==========
df = pd.read_csv('111.csv')
print(f"原始数据形状: {df.shape}")

# ========== 2. 薪资数据清洗 ==========
def parse_salary(salary_str):
    """解析薪资字符串，返回(最低薪, 最高薪)"""
    if pd.isna(salary_str):
        return None, None
    salary_str = str(salary_str).strip()
    # 匹配 "5-8千/月", "1.5-3万/月"
    match = re.match(r'([\d.]+)\s*[-~]\s*([\d.]+)\s*(万|千)/月', salary_str)
    if match:
        low = float(match.group(1))
        high = float(match.group(2))
        unit = match.group(3)
        if unit == '万':
            return low * 10000, high * 10000
        else:
            return low * 1000, high * 1000
    # 匹配 "5千/月"
    match = re.match(r'([\d.]+)\s*(万|千)/月', salary_str)
    if match:
        val = float(match.group(1))
        unit = match.group(2)
        if unit == '万':
            val *= 10000
        else:
            val *= 1000
        return val, val
    return None, None

df['最低薪资'] = df['薪资'].apply(lambda x: parse_salary(x)[0])
df['最高薪资'] = df['薪资'].apply(lambda x: parse_salary(x)[1])
df['平均薪资'] = df.apply(
    lambda row: (row['最低薪资'] + row['最高薪资']) / 2
    if not pd.isna(row['最低薪资']) and not pd.isna(row['最高薪资']) else np.nan, axis=1)

# 薪资区间分类
def classify_salary(avg):
    if pd.isna(avg): return '未知'
    if avg < 5000: return '5千以下'
    elif avg < 8000: return '5-8千'
    elif avg < 12000: return '8千-1.2万'
    elif avg < 20000: return '1.2-2万'
    elif avg < 30000: return '2-3万'
    else: return '3万以上'

df['薪资区间'] = df['平均薪资'].apply(classify_salary)

# ========== 3. 城市/公司规模/行业清洗 ==========
def extract_city(city_str):
    if pd.isna(city_str): return '未知'
    city_str = str(city_str).strip()
    match = re.match(r'([\u4e00-\u9fa5]+)-', city_str)
    if match: return match.group(1)
    match = re.match(r'([\u4e00-\u9fa5]+)', city_str)
    if match: return match.group(1)
    return city_str

df['城市'] = df['城市'].apply(extract_city)

def clean_list_str(s):
    if pd.isna(s): return ''
    s = str(s).strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    s = s.strip("'\"")
    return s.strip()

df['公司规模_clean'] = df['公司规模'].apply(clean_list_str)
df['公司类型_clean'] = df['公司类型'].apply(clean_list_str)

def clean_industry(s):
    if pd.isna(s): return ''
    s = str(s).strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    s = s.strip("'\"")
    if ',' in s: s = s.split(',')[0].strip()
    return s.strip()

df['行业_clean'] = df['行业'].apply(clean_industry)

# ========== 4. 统计分析 ==========
city_exclude = df[~df['城市'].isin(['异地招聘', '未知'])]

# 城市岗位分布 Top15
city_count = city_exclude['城市'].value_counts().head(15)
city_data = [{"城市": k, "岗位数量": int(v)} for k, v in city_count.items()]

# 行业分布 Top15
industry_count = df['行业_clean'].value_counts().head(15)
industry_data = [{"行业": k, "岗位数量": int(v)} for k, v in industry_count.items() if k]

# 薪资区间分布
salary_range_order = ['5千以下', '5-8千', '8千-1.2万', '1.2-2万', '2-3万', '3万以上', '未知']
salary_range_count = df['薪资区间'].value_counts()
salary_range_data = [{"薪资区间": sr, "岗位数量": int(salary_range_count.get(sr, 0))} for sr in salary_range_order]

# 各城市平均薪资 Top15
city_salary = city_exclude.groupby('城市')['平均薪资'].mean().dropna().sort_values(ascending=False).head(15)
city_salary_data = [{"城市": k, "平均薪资": round(float(v), 0)} for k, v in city_salary.items()]

# 公司规模分布
company_size_data = [{"公司规模": k, "岗位数量": int(v)} for k, v in df['公司规模_clean'].value_counts().head(10).items() if k]

# 公司类型分布
company_type_data = [{"公司类型": k, "岗位数量": int(v)} for k, v in df['公司类型_clean'].value_counts().head(10).items() if k]

# 岗位关键词
job_titles = df['岗位名称'].dropna().tolist()
keywords_list = ['数据', '分析', '算法', '产品', '运营', '开发', '设计', '市场', '销售', '管理',
                 '财务', '人事', '客服', '测试', '前端', '后端', '全栈', '大数据', '人工智能', '机器学习']
keywords = []
for title in job_titles:
    for kw in keywords_list:
        if kw in str(title):
            keywords.append(kw)
keyword_data = [{"关键词": k, "出现次数": int(v)} for k, v in Counter(keywords).most_common(15)]

# 岗位词频Top20
title_word_count = Counter()
for title in job_titles:
    words = re.findall(r'[\u4e00-\u9fa5]{2,}', str(title))
    for w in words:
        if w not in ['专员', '助理', '主管', '经理', '总监', '工程师', '分析师', '专家', '人才', '招聘']:
            title_word_count[w] += 1
title_word_data = [{"岗位词": k, "出现次数": int(v)} for k, v in title_word_count.most_common(20)]

# 城市×薪资区间交叉表
top_cities = city_exclude['城市'].value_counts().head(8).index.tolist()
city_mask = city_exclude['城市'].isin(top_cities)
city_cs = pd.crosstab(city_exclude[city_mask]['城市'], city_exclude[city_mask]['薪资区间'])
col_order = [s for s in salary_range_order if s in city_cs.columns]
city_cs = city_cs[[c for c in col_order if c in city_cs.columns]]
city_salary_range_data = {
    "categories": [str(c) for c in city_cs.columns.tolist()],
    "series": [{"name": str(r), "data": [int(v) for v in city_cs.loc[r].values]} for r in city_cs.index.tolist()]
}

# 各城市高薪岗位比例(>=2万)
city_high_salary = city_exclude.groupby('城市').apply(
    lambda g: (g['平均薪资'] >= 20000).sum() / g['平均薪资'].notna().sum() * 100
    if g['平均薪资'].notna().sum() > 0 else 0, include_groups=False
).sort_values(ascending=False).head(10)
city_high_salary_data = [{"城市": k, "高薪岗位比例": round(float(v), 1)} for k, v in city_high_salary.items()]

# ========== 5. 保存结果 ==========
analysis_results = {
    "city_count": city_data,
    "industry_count": industry_data,
    "salary_range_count": salary_range_data,
    "city_salary_avg": city_salary_data,
    "company_size_count": company_size_data,
    "company_type_count": company_type_data,
    "keyword_count": keyword_data,
    "title_word_count": title_word_data,
    "city_salary_range": city_salary_range_data,
    "city_high_salary": city_high_salary_data,
    "total_jobs": len(df),
    "valid_salary_count": int(df['平均薪资'].notna().sum()),
    "city_count_total": city_exclude['城市'].nunique(),
}

os.makedirs('output', exist_ok=True)
with open('output/recruitment_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(analysis_results, f, ensure_ascii=False, indent=2)

print(f"\n分析完成!")
print(f"总岗位数: {len(df)}")
print(f"有效薪资数: {df['平均薪资'].notna().sum()}")
print(f"城市数: {city_exclude['城市'].nunique()}")
print(f"结果已保存至 output/recruitment_analysis.json")