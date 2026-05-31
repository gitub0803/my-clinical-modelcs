import streamlit as st
import pandas as pd
import joblib
import os

# 1. 页面基本配置
st.set_page_config(page_title="单个患者临床风险评估系统", page_icon="🩺", layout="centered")

st.title("🩺 单个患者临床风险评估系统")
st.write("本系统已成功对接预后评估模型。请在下方输入单个患者的指标进行实时风险评估。")
st.divider()

# 2. 载入 joblib 保存的 pkl 模型
MODEL_PATH = "clinic_model.pkl" # 确保该文件与 app.py 在同一文件夹下

@st.cache_resource  # 缓存机制，避免重复加载
def load_saved_model(path):
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            return model
        except Exception as e:
            st.error(f"模型文件解析失败: {e}")
            return None
    return None

model = load_saved_model(MODEL_PATH)

if model is None:
    st.error(f"❌ 未在当前目录下找到模型文件 `{MODEL_PATH}`，请确保它已被复制到网页文件夹中！")
    st.stop()

# 提取模型期望的特征名称列表
try:
    expected_features = model.feature_names_in_
except AttributeError:
    expected_features = None

# 3. 单个患者输入表单界面
if expected_features is not None:
    st.subheader("📋 请输入当前患者临床指标")
    
    # 动态创建单人输入的表单
    input_data = {}
    
    # 两列并排摆放组件，让表单结构更紧凑美观
    cols = st.columns(2)
    for idx, feat in enumerate(expected_features):
        with cols[idx % 2]:
            # 这里默认初始值为 0.0，步长 0.01。你可以根据实际临床指标的范围修改初始值
            input_data[feat] = st.number_input(f"指标: {feat}", value=0.0, step=0.01)
            
    st.divider()
    
    # 4. 点击计算触发预测
    if st.button("🚀 开始计算该患者预后结局", type="primary", use_container_width=True):
        try:
            # 将输入数据转化为符合模型要求的 DataFrame，并强制保持特征顺序一致
            single_X = pd.DataFrame([input_data])[expected_features]
            
            # 计算单人发生结局的概率
            prob = model.predict_proba(single_X)[0][1]
            # 计算模型预测的分类（0 或 1）
            pred_class = model.predict(single_X)[0]
            
            # 展示评估结果
            st.subheader("📊 评估报告")
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric(label="预测发生风险 (Probability)", value=f"{prob*100:.2f}%")
            with c2:
                st.metric(label="模型预测最终结局 (Class)", value=int(pred_class))
                
            # 根据概率大小给出临床提示
            if prob > 0.5:
                st.error("⚠️ 提示：该患者预测发生结局的概率较高，临床上建议密切关注。")
            else:
                st.success("✅ 提示：该患者当前处于低风险预后范围。")
                
        except Exception as e:
            st.error(f"计算失败，请检查输入值是否合理。报错信息: {e}")
else:
    st.error("❌ 无法从模型中提取特征名称，请确保模型训练时使用的是带有列名的 DataFrame。")