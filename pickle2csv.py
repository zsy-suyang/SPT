import pandas as pd
import pickle

# === 路径定义 ===
input_pkl = "/internfs/zsy/proj/AMFormer/datasets/exp_xanes.pkl"
output_csv = "/internfs/zsy/proj/AMFormer/datasets/exp_xanes.csv"

# === 读取 pkl 文件 ===
with open(input_pkl, "rb") as f:
    data = pickle.load(f)

# === 判断数据类型 ===
if isinstance(data, pd.DataFrame):
    df = data
elif isinstance(data, dict):
    df = pd.DataFrame(data)
elif isinstance(data, list):
    try:
        df = pd.DataFrame(data)
    except Exception:
        df = pd.DataFrame([data])
else:
    raise TypeError(f"❌ 不支持的类型: {type(data)}")

df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"✅ 已成功将 {input_pkl} 转换为 {output_csv}")
print(f"📊 数据形状: {df.shape}, {df.columns}")
