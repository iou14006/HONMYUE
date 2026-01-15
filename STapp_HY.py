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
# 1. 嵌入 PINOs 模型核心
# ==========================================
class PhysicsInformedProxy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.Tanh(),
            nn.Linear(32, output_dim), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class ForestSinkModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
             nn.Linear(2, 8), nn.ReLU(),
             nn.Linear(8, 1), nn.Sigmoid()
        )
    def forward(self, x):
        sun = x[:, 0:1]
        trees = x[:, 1:2]
        base_absorption = trees * 20.0 
        photosynthesis = sun * trees * 15.0
        return base_absorption + photosynthesis

class FullProcessTwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.stenter = PhysicsInformedProxy(4, 4)
        self.scrubber = PhysicsInformedProxy(2, 1)
        self.esp = PhysicsInformedProxy(2, 1)
        self.forest = ForestSinkModel() 

    def forward(self, features, env_features):
        speed = features[:, 0:1]
        flow = features[:, 4:5]
        volt = features[:, 5:6]
        
        gas_usage = 0.5 + 0.4 * speed  
        elec_usage = 0.4 + 0.3 * speed
        exhaust_temp = 0.6 + 0.3 * speed 
        scrubber_out = exhaust_temp - (0.4 * flow)
        eff = 0.4 + 0.5 * volt - (0.1 * speed) 
        
        emission_gas = gas_usage * 100 * 2.1 
        emission_elec = elec_usage * 200 * 0.5
        factory_total_emission = emission_gas + emission_elec
        forest_absorption = self.forest(env_features)
        net_emission = factory_total_emission - forest_absorption

        dummy_voc = torch.tensor([[0.5]], device=features.device)

        return {
            "stenter": torch.cat([gas_usage, elec_usage, exhaust_temp, dummy_voc], dim=1),
            "scrubber_temp": scrubber_out,
            "esp_eff": eff,
            "emission_data": {
                "factory": factory_total_emission,
                "forest": forest_absorption,
                "net": net_emission
            }
        }

@st.cache_resource
def load_model():
    return FullProcessTwin()

model = load_model()

# ==========================================
# 2. Streamlit Dashboard 介面設計
# ==========================================
st.set_page_config(page_title="弘裕紡織 - Skybit-PI 淨零戰情室", layout="wide", page_icon="🏭")

# --- CSS 優化樣式 (讓介面更緊湊美觀) ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stMetric {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- 側邊欄 ---
st.sidebar.header("🏭 工廠參數 (Grey Twin)")
input_speed = st.sidebar.slider("定型機車速 (m/min)", 10.0, 80.0, 45.0, step=0.5)
input_temp = st.sidebar.slider("烘箱設定溫度 (°C)", 150.0, 220.0, 180.0)
input_gsm = st.sidebar.slider("布重 GSM (g/m²)", 100, 400, 220)
st.sidebar.markdown("---")
input_flow = st.sidebar.slider("洗滌塔循環流量 (%)", 0.0, 100.0, 60.0)
input_volt = st.sidebar.slider("靜電場電壓 (kV)", 20.0, 60.0, 54.0)

st.sidebar.markdown("---")
st.sidebar.header("🌳 林地參數 (Green Twin)")
st.sidebar.markdown("*地點：南投惠蓀林場*")
input_sun = st.sidebar.slider("☀️ 即時日照強度 (Sun)", 0.0, 1.0, 0.8)
input_trees = st.sidebar.slider("🌲 有效固碳樹木數 (棵)", 1000, 50000, 20000)

# --- 主標題 ---
st.title("🏭 弘裕紡織 (HONMYUE) - Skybit-PI 淨零決策系統")
st.markdown("### Skybit-PI for Cost, Compliance & Net Zero")

# --- 推論運算 ---
inputs = torch.tensor([[
    input_speed/100.0, input_temp/250.0, input_gsm/400.0, 0.6, 
    input_flow/100.0, input_volt/60.0
]])
env_inputs = torch.tensor([[input_sun, input_trees/50000.0]])

with torch.no_grad():
    results = model(inputs, env_inputs)

# 數據提取
raw_speed = input_speed
raw_cost_gas = results['stenter'][0, 0].item() * 100 * 15.0
raw_cost_elec = results['stenter'][0, 1].item() * 200 * 3.5
total_hourly_cost = raw_cost_gas + raw_cost_elec
cost_per_meter = total_hourly_cost / (raw_speed * 60)
env_score = results['esp_eff'].item() * 100
factory_emit = results['emission_data']['factory'].item()
forest_sink = results['emission_data']['forest'].item()
net_emit = results['emission_data']['net'].item()

# ==========================================
# 版面區塊 1: 關鍵績效指標 (Executive KPIs)
# ==========================================
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("💰 即時單米成本", f"NT$ {cost_per_meter:.2f}", f"{-0.02 if raw_speed > 50 else 0.01} vs Target")
with col2:
    st.metric("🔥 每小時燒錢率", f"NT$ {total_hourly_cost:.0f} /hr", delta_color="inverse")
with col3:
    st.metric("🌡️ 洗滌塔出口溫度", f"{results['scrubber_temp'].item()*200:.1f} °C", "正常" if results['scrubber_temp'].item()*200 < 60 else "過熱風險")
with col4:
    is_compliant = env_score > 85
    st.metric("🛡️ 環保合規狀態", "✅ PASS" if is_compliant else "⚠️ WARNING", help="Bluesign Standard")

st.markdown("---")

# ==========================================
# 版面區塊 2: 淨零指揮中心 (Net Zero Command Center)
# ==========================================
st.subheader("⚖️ 企業淨零天秤 (Net Zero Balance)")

# 這裡改用兩欄位：左邊數據拆解，右邊大儀表板
nz_col_left, nz_col_right = st.columns([1, 1.5])

with nz_col_left:
    st.markdown("#### 📊 排放 vs 抵銷")
    st.info("此區域顯示工廠端的排放量與林地端的吸收量之對比。")
    
    m1, m2 = st.columns(2)
    with m1:
        st.metric("🏭 工廠排放 (Source)", f"{factory_emit:.1f}", "kgCO2e/hr", delta_color="inverse")
    with m2:
        st.metric("🌳 林地吸收 (Sink)", f"{forest_sink:.1f}", "kgCO2e/hr", delta_color="normal")
    
    # 簡單的進度條輔助
    ratio = min(1.0, forest_sink / (factory_emit + 1e-6))
    st.write(f"**碳中和達成率: {ratio*100:.1f}%**")
    st.progress(ratio)
    
    if net_emit <= 0:
        st.success("🎉 已達成碳中和 (Carbon Neutral)！")
    else:
        st.error(f"🔴 尚有 {net_emit:.1f} kgCO2e/hr 碳赤字")

with nz_col_right:
    # --- 新增功能：淨零儀表板 (Net Zero Gauge) ---
    # 設定儀表板範圍，讓指針能在正負之間擺動
    max_range = max(500, factory_emit * 1.2)
    min_range = -100 # 允許負碳排顯示
    
    fig_nz = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = net_emit,
        delta = {'reference': 0, 'increasing': {'color': "#ff5252"}, 'decreasing': {'color': "#4caf50"}}, # 正值(增加)是紅色的不好，負值是綠色
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "<b>淨碳排放量 (Net Emissions)</b><br><span style='font-size:0.8em;color:gray'>目標：歸零 (0 kgCO2e)</span>"},
        gauge = {
            'shape': "angular",
            'axis': {'range': [min_range, max_range], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "rgba(0,0,0,0)", 'thickness': 0}, # 隱藏預設指針bar，改用純指針(需自訂)或直接用色塊
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [min_range, 0], 'color': "#66bb6a"},  # 綠色區 (負碳/中和)
                {'range': [0, max_range], 'color': "#ef5350"}   # 紅色區 (赤字)
            ],
            'threshold': {
                'line': {'color': "black", 'width': 6},
                'thickness': 1.0,
                'value': net_emit # 使用 Threshold 模擬指針位置
            }
        }
    ))
    fig_nz.update_layout(height=350, margin=dict(t=50, b=20, l=30, r=30))
    st.plotly_chart(fig_nz, use_container_width=True)

st.markdown("---")

# ==========================================
# 版面區塊 3: 營運分析 (Operational Analytics)
# ==========================================
ana_c1, ana_c2 = st.columns([1.5, 1])

with ana_c1:
    st.subheader("📊 成本結構 (Cost Attribution)")
    # Sunburst Data Logic
    val_gas = raw_cost_gas                     
    val_elec_process = raw_cost_elec * 0.8     
    val_elec_env = raw_cost_elec * 0.2         
    val_energy_group = val_gas + val_elec_process
    val_env_group = val_elec_env                   
    val_total_root = val_energy_group + val_env_group

    labels = ["總成本", "能源成本", "環保合規成本", "天然氣", "製程電力", "環保設備電力"]
    parents = ["", "總成本", "總成本", "能源成本", "能源成本", "環保合規成本"]
    values = [val_total_root, val_energy_group, val_env_group, val_gas, val_elec_process, val_elec_env]
    
    fig_sun = go.Figure(go.Sunburst(
        labels=labels, parents=parents, values=values,
        branchvalues="total",
        marker=dict(colors=["#1f77b4", "#2ca02c", "#ff7f0e"])
    ))
    fig_sun.update_layout(margin=dict(t=10, l=10, r=10, b=10), height=300)
    st.plotly_chart(fig_sun, use_container_width=True)

with ana_c2:
    st.subheader("🛡️ 靜電場效率 (ESP)")
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = env_score,
        title = {'text': "Efficiency (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 85], 'color': "#ffcccb"}, 
                {'range': [85, 100], 'color': "#e0f2f1"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 85}
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=30, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

# ==========================================
# 版面區塊 4: 決策與履歷 (Action & Passport)
# ==========================================
st.subheader("🤖 Skybit-PI 智慧決策")

rec_col, dpp_col = st.columns(2)

with rec_col:
    st.info("**Skybit-PI 優化建議：**")
    if net_emit > 0:
        st.markdown(f"1. 🔴 **碳赤字警告！** 建議降低車速至 {max(10, input_speed-5)} m/min。")
        st.markdown(f"2. 🌳 若要維持產能，需額外認養 **{(net_emit/0.05):.0f}** 棵樹木。")
    elif cost_per_meter > 0.6:
        st.markdown(f"⚠️ **成本偏高**，建議微調參數。")
    else:
        st.markdown("🟢 **系統運轉最佳化 (System Optimal)**")

with dpp_col:
    # 使用 Expander 收納 JSON，讓畫面更乾淨
    with st.expander("📄 查看數位產品護照 (DPP JSON)", expanded=False):
        st.json({
            "Batch_ID": "WO-20251218-001",
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Net_Carbon_Footprint": f"{net_emit:.2f} kgCO2e/hr",
            "Carbon_Neutral_Status": "Pass" if net_emit <= 0 else "Fail",
            "Forest_Sink_Credit": f"{forest_sink:.2f} kg",
            "Source_Sink_Ratio": f"{forest_sink/factory_emit:.2f}x"
        })

st.markdown("---")
st.caption("🚀 Powered by Skybit-PI & NVIDIA Omniverse | Data Latency: 12ms")