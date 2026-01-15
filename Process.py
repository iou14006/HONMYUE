import torch
import torch.nn as nn
import json
import datetime
import uuid

# ==========================================
# 0. 基礎模型骨幹 (共用)
# ==========================================
class PhysicsInformedProxy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(PhysicsInformedProxy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# ==========================================
# 1. 定義各單元 PINOs 模型
# ==========================================

# --- A. 定型機模型 (Stenter) ---
# 輸入: [車速, 設定溫度, GSM, 含水率] (4 features)
# 輸出: [天然氣耗量, 電耗, 排氣溫度, 排氣VOC] (4 features)
class StenterModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Input=4, Output=4
        self.model = PhysicsInformedProxy(input_dim=4, output_dim=4)
    
    def forward(self, x):
        return self.model(x)

# --- B. 洗滌塔模型 (Scrubber) ---
# 輸入: [進氣溫度, 水流量] (2 features)
# 輸出: [出口溫度] (1 feature)
class ScrubberModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Input=2, Output=1
        self.model = PhysicsInformedProxy(input_dim=2, output_dim=1)
    
    def forward(self, x):
        return self.model(x)

# --- C. 靜電場模型 (ESP) ---
# 輸入: [累積運行時間, 電壓] (2 features)
# 輸出: [除油效率] (1 feature)
class ESPModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Input=2, Output=1
        self.model = PhysicsInformedProxy(input_dim=2, output_dim=1)

    def forward(self, x):
        return self.model(x)

# ==========================================
# 2. 全製程串聯模型 (Process Orchestrator)
# ==========================================
class FullProcessTwin(nn.Module):
    def __init__(self, stenter, scrubber, esp):
        super().__init__()
        self.stenter = stenter
        self.scrubber = scrubber 
        self.esp = esp            
        
        # --- 成本參數 (可從 ERP 讀取) ---
        self.unit_price = {
            "gas_per_m3": 15.0,    # 天然氣單價
            "elec_per_kwh": 3.5,   # 電費單價
            "water_per_ton": 12.0, # 水費單價
            "chem_per_kg": 25.0    # 洗滌塔藥劑單價
        }

    def forward(self, work_order_features):
        """
        work_order_features: [Speed, Temp_Set, GSM, Moisture_In, Flow_Scrubber, Volt_ESP]
        """
        # 1. 定型機推論 (First Stage)
        # 輸入前4個特徵
        stenter_in = work_order_features[:, 0:4]
        stenter_out = self.stenter(stenter_in)
        
        # 解包定型機輸出
        gas_rate = stenter_out[:, 0:1]
        elec_stenter = stenter_out[:, 1:2]
        exhaust_temp = stenter_out[:, 2:3] # -> 傳給洗滌塔
        exhaust_voc = stenter_out[:, 3:4]

        # 2. 物理耦合: 定型機排氣 -> 洗滌塔進氣
        # 洗滌塔輸入需要 2 個特徵: [排氣溫度, 水流量]
        scrubber_flow_setting = work_order_features[:, 4:5]
        
        # 這裡將 [1x1] 和 [1x1] 拼接成 [1x2]，符合 ScrubberModel 的輸入要求
        scrubber_in = torch.cat([exhaust_temp, scrubber_flow_setting], dim=1) 
        
        scrubber_out_temp = self.scrubber(scrubber_in) 

        # 3. 物理耦合: 洗滌塔出口 -> 靜電機入口
        # 靜電機輸入需要 2 個特徵: [運行時間, 電壓]
        esp_time_accum = torch.tensor([[0.5]]) # 模擬運行中
        esp_volt_setting = work_order_features[:, 5:6]
        
        esp_in = torch.cat([esp_time_accum, esp_volt_setting], dim=1)
        
        esp_efficiency = self.esp(esp_in)

        return {
            "stenter": stenter_out,
            "scrubber_temp": scrubber_out_temp,
            "esp_eff": esp_efficiency
        }

# ==========================================
# 3. 成本歸屬與標準化輸出模組
# ==========================================
class CostAccountingInterface:
    def __init__(self, full_model, equipment_id="Line-01"):
        self.model = full_model
        self.equipment_id = equipment_id

    def generate_full_report(self, work_order_data):
        # 將字典轉為 Tensor
        inputs = torch.tensor([[
            work_order_data['speed_m_min'] / 100.0,    # 歸一化
            work_order_data['temp_set_c'] / 250.0,
            work_order_data['gsm'] / 400.0,
            work_order_data['moisture_in'] / 100.0,
            work_order_data['scrubber_flow_ratio'],
            work_order_data['esp_voltage_ratio']
        ]])

        # 執行全製程推論
        with torch.no_grad():
            results = self.model(inputs)

        # --- 反歸一化與成本計算 ---
        speed = work_order_data['speed_m_min'] 
        width = work_order_data['width_m']     
        
        # 物理消耗還原 (模擬值)
        gas_usage_m3_h = results['stenter'][0, 0].item() * 100.0 
        elec_total_kw = results['stenter'][0, 1].item() * 200.0  
        
        # 每小時總成本
        cost_gas = gas_usage_m3_h * self.model.unit_price['gas_per_m3']
        cost_elec = elec_total_kw * self.model.unit_price['elec_per_kwh']
        cost_env = (elec_total_kw * 0.15) * self.model.unit_price['elec_per_kwh'] 
        
        total_hourly_cost = cost_gas + cost_elec
        
        # 每米成本
        cost_per_meter = total_hourly_cost / (speed * 60.0)

        # JSON 結構組裝
        payload = {
            "meta": {
                "wo_id": work_order_data['wo_id'],
                "material_id": work_order_data['material_id'],
                "timestamp": datetime.datetime.now().isoformat(),
                "batch_status": "Processing"
            },
            "production_dynamics": {
                "current_speed_m_min": speed,
                "current_width_m": width,
                "process_stage": "Stenter-Curing"
            },
            "physics_metrics": {
                "stenter_exhaust_temp_c": round(results['stenter'][0, 2].item() * 200, 1),
                "scrubber_outlet_temp_c": round(results['scrubber_temp'].item() * 200, 1),
                "esp_removal_efficiency": round(results['esp_eff'].item(), 4)
            },
            "cost_allocation": {
                "currency": "TWD",
                "realtime_burn_rate_hr": round(total_hourly_cost, 2),
                "unit_cost_breakdown": {
                    "per_meter_total": round(cost_per_meter, 4),
                    "per_meter_energy": round((cost_gas + cost_elec)/ (speed * 60), 4),
                    "per_meter_environmental": round(cost_env / (speed * 60), 4)
                }
            },
            "omniverse_integration": {
                "cost_floating_ui": f"NT$ {cost_per_meter:.2f} / m",
                "machine_color_state": "Running" if speed > 0 else "Idle"
            }
        }
        return payload

# ==========================================
# 4. 主程式執行區塊
# ==========================================
if __name__ == "__main__":
    # 1. 正確實例化三個不同的模型
    stenter_model = StenterModel()    # Input dim: 4
    scrubber_model = ScrubberModel()  # Input dim: 2 (修正點)
    esp_model = ESPModel()            # Input dim: 2 (修正點)

    # 2. 注入到全製程孿生物件中
    full_system = FullProcessTwin(stenter=stenter_model, scrubber=scrubber_model, esp=esp_model) 
    cost_interface = CostAccountingInterface(full_system)

    # 3. 模擬 ERP 傳來的工單資料
    wo_data = {
        "wo_id": "WO-20251218-001",
        "material_id": "Recycled-PET-Fabric-TypeA",
        "speed_m_min": 45.0,    
        "width_m": 1.6,
        "temp_set_c": 180.0,
        "gsm": 220,
        "moisture_in": 60,      
        "scrubber_flow_ratio": 0.6,
        "esp_voltage_ratio": 0.9
    }

    # 4. 輸出結果
    print("--- 弘裕紡織 PINOs 智慧成本運算系統 ---")
    print(json.dumps(cost_interface.generate_full_report(wo_data), indent=4, ensure_ascii=False))