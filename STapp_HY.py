import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import numpy as np

# =========================================================
# DEVICE SELECTOR
# =========================================================
SUPPORTED_CC = {(5, 0), (6, 0), (6, 1), (7, 0), (7, 5), (8, 0), (8, 6), (9, 0)}

def safe_pick_device():
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            if (major, minor) not in SUPPORTED_CC:
                return torch.device("cpu")
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")

DEVICE = safe_pick_device()

# ==========================================
# 1. 嵌入 PINOs 模型核心 (Mock for Demo)
#    (實際部署時可直接 import Process)
# ==========================================
class PhysicsInformedProxy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.Tanh(),
            nn.Linear(32, output_dim), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class FullProcessTwin(nn.Module):
    def __init__(self):
        super().__init__()
        # 為了 Demo 方便，這些子模組在 forward 中暫時用模擬邏輯取代，但保留定義以免報錯
        self.stenter = PhysicsInformedProxy(4, 4)
        self.scrubber = PhysicsInformedProxy(2, 1)
        self.esp = PhysicsInformedProxy(2, 1)
        self.unit_price = {
            "gas_per_m3": 15.0, "elec_per_kwh": 3.5, "water_per_ton": 12.0
        }

    def forward(self, features):
        # features: [Speed, Temp, GSM, Moisture, Scrubber_Flow, ESP_Volt]
        
        # --- 修正點開始：使用 [:, i:i+1] 語法來保持 2D 維度 ---
        speed = features[:, 0:1]       # Shape 變為 (Batch, 1) 而非 (Batch,)
        temp = features[:, 1:2]
        # flow = features[:, 4:5]      # 若需要 flow 計算也請保持這樣切
        # volt = features[:, 5:6]
        
        # 為了配合下方的計算邏輯，我們確保變數都是 2D Tensor
        flow = features[:, 4:5]
        volt = features[:, 5:6]
        # --- 修正點結束 ---

        # 1. 定型機模擬邏輯
        gas_usage = 0.5 + 0.4 * speed  
        elec_usage = 0.4 + 0.3 * speed
        exhaust_temp = 0.6 + 0.3 * speed 
        
        # 2. 洗滌塔模擬邏輯
        scrubber_out = exhaust_temp - (0.4 * flow)
        
        # 3. 靜電場模擬邏輯
        eff = 0.4 + 0.5 * volt - (0.1 * speed) 
        
        # 構建常數 Tensor 時，確保維度與設備匹配
        dummy_voc = torch.tensor([[0.5]], device=features.device, dtype=features.dtype)

        return {
            "stenter": torch.cat([gas_usage, elec_usage, exhaust_temp, dummy_voc], dim=1),
            "scrubber_temp": scrubber_out,
            "esp_eff": eff
        }

# 初始化模型
@st.cache_resource
def load_model():
    return FullProcessTwin()

model = load_model()

# ==========================================
# 2. Streamlit Dashboard 介面設計
# ==========================================
st.set_page_config(page_title="弘裕紡織 - 智慧成本戰情室", layout="wide", page_icon="🏭")

# --- 側邊欄：數位分身控制台 (Digital Twin Controller) ---
st.sidebar.header("🎛️ 數位分身參數控制")
st.sidebar.markdown("模擬調整現場參數，即時預測成本與合規性。")

input_speed = st.sidebar.slider("定型機車速 (m/min)", 10.0, 80.0, 45.0, step=0.5)
input_temp = st.sidebar.slider("烘箱設定溫度 (°C)", 150.0, 220.0, 180.0)
input_gsm = st.sidebar.slider("布重 GSM (g/m²)", 100, 400, 220)
st.sidebar.markdown("---")
input_flow = st.sidebar.slider("洗滌塔循環流量 (%)", 0.0, 100.0, 60.0)
input_volt = st.sidebar.slider("靜電場電壓 (kV)", 20.0, 60.0, 54.0)

# --- 主畫面 ---
st.title("🏭 弘裕紡織 (Hongyu Textile) - PINOs 智慧決策系統")
st.markdown("### Process-Informed Neural Operators for Cost & Compliance")

# 準備輸入數據
inputs = torch.tensor([[
    input_speed/100.0, input_temp/250.0, input_gsm/400.0, 0.6, 
    input_flow/100.0, input_volt/60.0
]])

# 執行推論
with torch.no_grad():
    results = model(inputs)

# 計算顯示數據
raw_speed = input_speed
raw_cost_gas = results['stenter'][0, 0].item() * 100 * 15.0
raw_cost_elec = results['stenter'][0, 1].item() * 200 * 3.5
total_hourly_cost = raw_cost_gas + raw_cost_elec
cost_per_meter = total_hourly_cost / (raw_speed * 60)
env_score = results['esp_eff'].item() * 100

# --- 第一排：關鍵 KPI (North Star Metrics) ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="💰 即時單米成本 (Cost/m)", 
              value=f"NT$ {cost_per_meter:.2f}", 
              delta=f"{-0.02 if raw_speed > 50 else 0.01} vs Target")
    
with col2:
    st.metric(label="🔥 每小時燒錢率 (Burn Rate)", 
              value=f"NT$ {total_hourly_cost:.0f} /hr",
              delta_color="inverse")

with col3:
    st.metric(label="🌡️ 洗滌塔出口溫度", 
              value=f"{results['scrubber_temp'].item()*200:.1f} °C",
              delta="正常" if results['scrubber_temp'].item()*200 < 60 else "過熱風險")

with col4:
    # 合規狀態邏輯
    is_compliant = env_score > 85
    status_text = "✅ PASS (符合 bluesign)" if is_compliant else "⚠️ WARNING"
    st.metric(label="🛡️ GRS/環保合規狀態", value=status_text)

st.markdown("---")

# --- 第二排：圖表區 ---
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("📊 即時成本結構分析 (Cost Attribution)")
    
    # --- 修正開始：確保父子層級數值邏輯一致 (由下往上加總) ---
    # 1. 先算出最底層的子項目 (Leaf Nodes)
    val_gas = raw_cost_gas                     # 天然氣
    val_elec_process = raw_cost_elec * 0.8     # 製程電力 (假設佔 80%)
    val_elec_env = raw_cost_elec * 0.2         # 環保電力 (假設佔 20%)
    
    # 2. 再算出中間層的父項目 (Parent Nodes)
    val_energy_group = val_gas + val_elec_process  # 能源成本群組
    val_env_group = val_elec_env                   # 環保成本群組
    
    # 3. 最後算出總成本 (Root Node)
    val_total_root = val_energy_group + val_env_group

    # 4. 建立圖表數據
    labels = ["總成本", "能源成本", "環保合規成本", "天然氣", "製程電力", "環保設備電力"]
    parents = ["", "總成本", "總成本", "能源成本", "能源成本", "環保合規成本"]
    values = [
        val_total_root,    # Root
        val_energy_group,  # Parent A
        val_env_group,     # Parent B
        val_gas,           # Child of A
        val_elec_process,  # Child of A
        val_elec_env       # Child of B
    ]
    # --- 修正結束 ---
    
    fig_sun = go.Figure(go.Sunburst(
        labels=labels, parents=parents, values=values,
        branchvalues="total",
        marker=dict(colors=["#1f77b4", "#2ca02c", "#ff7f0e"])
    ))
    fig_sun.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=300)
    st.plotly_chart(fig_sun, use_container_width=True)

with c2:
    st.subheader("🛡️ 靜電場除油效率")
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = env_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Efficiency (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 80], 'color': "#ffcccb"},  # 紅色警戒區
                {'range': [80, 100], 'color': "#90ee90"} # 綠色安全區
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 85
            }
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=30, b=0))
    st.plotly_chart(fig_gauge, use_container_width=True)

# --- 第三排：AI 建議與 GRS 追蹤 ---
st.subheader("🤖 PINOs AI 決策建議 (Prescriptive Analytics)")

col_rec1, col_rec2 = st.columns(2)

with col_rec1:
    st.info("**製程優化建議：**")
    if cost_per_meter > 0.6:
        st.markdown(f"🔴 目前單米成本過高！建議 **提升車速至 {input_speed + 5} m/min** 以分攤固定能耗。")
    elif env_score < 85:
        st.markdown("🔴 環保效率不足！建議 **增加靜電電壓** 或 **降低車速** 以符合排放標準。")
    else:
        st.markdown("🟢 目前運轉處於最佳甜蜜點 (Sweet Spot)。")

with col_rec2:
    st.success("**GRS 數據履歷 (區塊鏈預備)：**")
    st.json({
        "Batch_ID": "WO-20251218-001",
        "Carbon_Footprint": f"{total_hourly_cost/1000 * 0.5:.2f} kgCO2e/hr",
        "Heat_Recovery": "Enabled (Active)",
        "Water_Recycle_Rate": "85%"
    })

# --- Omniverse 連結示意 ---
st.markdown("---")
st.caption("🚀 Data Stream Status: Connected to NVIDIA Omniverse Nucleus | Protocol: USD/JSON | Latency: 12ms")