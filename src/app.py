import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import plotly.express as px
import plotly.graph_objects as go

# 1. Page Config
st.set_page_config(page_title="Kepler Exoplanet Classifier", page_icon="🔭", layout="wide")

# 2. Custom CSS
st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 2.5rem; color: #1E90FF; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { font-size: 1.2rem; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# 3. Header
st.markdown("<h1 style='text-align: center; color: #1E90FF;'>Kepler Exoplanet Classifier</h1>", unsafe_allow_html=True)
st.markdown("---")

# 3.1 Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# 4. Preprocessing Function (Robust for Batch)
def prepare_input(raw_data):
    d = raw_data.copy()
    
    # [FIX] เพิ่ม koi_dor เข้า required_raw_cols ด้วย
    required_raw_cols = ['koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_model_snr', 'koi_teq', 
                         'koi_srad', 'koi_steff', 'koi_slogg', 'koi_insol', 'koi_kepmag', 'koi_ror', 'koi_impact', 
                         'koi_srho', 'koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
                         'koi_dor']
    for c in required_raw_cols:
        if c not in d.columns:
            d[c] = 0.0

    log_cols = ['koi_depth', 'koi_prad', 'koi_insol', 'koi_period', 'koi_model_snr', 'koi_duration', 'koi_srho', 'koi_dor']
    for c in log_cols:
        if c in d.columns:
            d[f'log_{c}'] = np.log1p(d[c].clip(lower=0))
        else:
            d[f'log_{c}'] = 0.0

    d['log_transit_shape'] = np.log1p((d['koi_depth'] / (d['koi_duration'] ** 2 + 1e-6)).clip(lower=0))
    d['prad_srad_ratio'] = d['koi_prad'] / (d['koi_srad'] + 1e-6)
    d['log_prad_srad'] = np.log1p(d['prad_srad_ratio'].clip(lower=0))
    d['teq_steff_ratio'] = d['koi_teq'] / (d['koi_steff'] + 1e-6)
    d['log_snr_per_depth'] = np.log1p((d['koi_model_snr'] / (d['koi_depth'] + 1e-6)).clip(lower=0))
    d['log_slogg'] = np.log1p(d['koi_slogg'].clip(lower=0))

    # [FIX] ย้าย log_ror ขึ้นมาก่อน interaction terms ที่ใช้มัน
    d['log_ror'] = np.log1p(d['koi_ror'].clip(lower=0))

    d['snr_x_depth'] = d['log_koi_model_snr'] * d['log_koi_depth']
    d['insol_x_prad'] = d['log_koi_insol'] * d['log_koi_prad']
    d['period_x_depth'] = d['log_koi_period'] * d['log_koi_depth']
    d['prad_x_snr'] = d['log_koi_prad'] * d['log_koi_model_snr']

    fp_flags = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']
    d['fpflag_sum'] = d[fp_flags].sum(axis=1)
    d['fpflag_any'] = (d['fpflag_sum'] > 0).astype(int)

    d['impact_sq'] = d['koi_impact'] ** 2
    d['srho_x_ror'] = d['log_koi_srho'] * d['log_ror']

    err_pairs = ['koi_period', 'koi_depth', 'koi_prad', 'koi_duration', 'koi_ror', 'koi_srho', 'koi_impact']
    for val_col in err_pairs:
        d[f'uncert_{val_col}'] = np.log1p(0.001)

    d['koi_score_logit'] = np.log(d['koi_score'].clip(1e-4, 1-1e-4) / (1 - d['koi_score'].clip(1e-4, 1-1e-4)))

    d['period_bin'] = pd.cut(d['koi_period'], bins=[0, 5, 15, 50, 200, np.inf], labels=[0,1,2,3,4]).astype(float)
    d['size_cat'] = pd.cut(d['koi_prad'], bins=[0, 1.25, 2, 6, 15, np.inf], labels=[0,1,2,3,4]).astype(float)
    d['snr_bucket'] = pd.cut(d['koi_model_snr'], bins=[0, 10, 30, 100, np.inf], labels=[0,1,2,3]).astype(float)

    final_features = [
        'log_koi_depth', 'log_koi_prad', 'log_koi_insol', 'log_koi_period', 'log_koi_model_snr', 'log_koi_duration', 
        'log_koi_srho', 'log_koi_dor', 'log_transit_shape', 'log_prad_srad', 'teq_steff_ratio', 'log_snr_per_depth', 
        'log_slogg', 'snr_x_depth', 'insol_x_prad', 'period_x_depth', 'prad_x_snr', 'koi_fpflag_nt', 'koi_fpflag_ss', 
        'koi_fpflag_co', 'koi_fpflag_ec', 'fpflag_sum', 'fpflag_any', 'log_ror', 'impact_sq', 'srho_x_ror', 
        'koi_impact', 'koi_sma', 'koi_incl', 'uncert_koi_period', 'uncert_koi_depth', 'uncert_koi_prad', 
        'uncert_koi_duration', 'uncert_koi_ror', 'uncert_koi_srho', 'uncert_koi_impact', 'koi_score', 
        'koi_score_logit', 'period_bin', 'size_cat', 'snr_bucket', 'koi_teq', 'koi_srad', 'koi_steff', 
        'koi_slogg', 'koi_kepmag'
    ]
    
    for col in final_features:
        if col not in d.columns:
            d[col] = 0.0
    return d[final_features].fillna(0)

# 7. Reasoning & Visualization Components

def get_feature_scores(data, model):
    """
    คำนวณ feature contribution โดยใช้ feature_importances_ จาก model จริง
    แล้ว weight ด้วยค่าจริงของ feature นั้นๆ เพื่อให้ได้ per-sample score
    """
    feature_groups = {
        'Transit Signal (SNR)':   ['log_koi_model_snr', 'snr_x_depth', 'log_snr_per_depth', 'snr_bucket'],
        'Planet Size':            ['log_koi_prad', 'log_prad_srad', 'size_cat', 'impact_sq'],
        'Orbital Period':         ['log_koi_period', 'period_x_depth', 'period_bin'],
        'Transit Shape':          ['log_koi_depth', 'log_transit_shape', 'log_koi_duration'],
        'False Positive Flags':   ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'fpflag_sum', 'fpflag_any'],
        'Stellar Properties':     ['log_slogg', 'teq_steff_ratio', 'koi_steff', 'koi_srad'],
        'Orbit Geometry':         ['log_ror', 'srho_x_ror', 'koi_impact', 'log_koi_srho'],
    }

    scores = {}
    try:
        # ดึง feature importance จาก model จริง
        base_model = model
        if hasattr(model, 'named_steps'):
            for step in model.named_steps.values():
                if hasattr(step, 'feature_importances_'):
                    base_model = step
                    break
        
        if hasattr(base_model, 'feature_importances_'):
            importances = base_model.feature_importances_
            prepared = prepare_input(data)
            feat_names = prepared.columns.tolist()
            feat_vals = prepared.iloc[0].values

            for group, feats in feature_groups.items():
                total = 0.0
                for f in feats:
                    if f in feat_names:
                        idx = feat_names.index(f)
                        # importance * normalized feature value
                        total += importances[idx] * abs(float(feat_vals[idx]))
                scores[group] = total
        else:
            raise ValueError("No feature_importances_")
    except:
        # Fallback: rule-based scoring ถ้า model ไม่มี feature_importances_
        snr    = float(data['koi_model_snr'].iloc[0])
        p_rad  = float(data['koi_prad'].iloc[0])
        fp_sum = int(data['koi_fpflag_nt'].iloc[0] + data['koi_fpflag_ss'].iloc[0] +
                     data['koi_fpflag_co'].iloc[0] + data['koi_fpflag_ec'].iloc[0])
        depth  = float(data['koi_depth'].iloc[0])
        period = float(data['koi_period'].iloc[0])
        impact = float(data['koi_impact'].iloc[0])
        slogg  = float(data['koi_slogg'].iloc[0])

        scores = {
            'Transit Signal (SNR)':  min(snr / 100, 1.0),
            'Planet Size':           min(p_rad / 10, 1.0),
            'Orbital Period':        min(1 / (period + 1), 1.0),
            'Transit Shape':         min(np.log1p(depth) / 10, 1.0),
            'False Positive Flags':  max(0, 1.0 - fp_sum * 0.4),
            'Stellar Properties':    min(slogg / 5, 1.0),
            'Orbit Geometry':        max(0, 1.0 - impact),
        }

    # Normalize ให้ sum = 1
    total = sum(scores.values()) + 1e-9
    return {k: v / total for k, v in scores.items()}


def render_planet_scale_svg(p_rad, is_verdict):
    """SVG เปรียบเทียบขนาดดาวกับโลก — scale ถูกต้อง, มี label"""
    verdict_color = {'CONFIRMED': '#00CC96', 'CANDIDATE': '#FFC107', 'FALSE POSITIVE': '#EF553B'}
    planet_color = verdict_color.get(is_verdict, '#888')

    SVG_W, SVG_H = 280, 120
    cx_earth = 70
    cy = SVG_H // 2
    r_earth = 18  # radius ของ Earth ใน pixel (fixed reference)

    # Scale ดาว — log scale เพื่อไม่ให้ดาวยักษ์กินพื้นที่หมด
    r_planet_raw = r_earth * (p_rad ** 0.6)
    r_planet = min(max(r_planet_raw, 4), 54)  # clamp 4–54px

    cx_planet = SVG_W - 70

    # เลือก gradient color ของโลก
    earth_grad = "url(#earthGrad)"
    planet_grad = "url(#planetGrad)"

    svg = f"""
    <svg width="{SVG_W}" height="{SVG_H}" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="earthGrad" cx="35%" cy="35%">
          <stop offset="0%" stop-color="#6dd5fa"/>
          <stop offset="100%" stop-color="#2980b9"/>
        </radialGradient>
        <radialGradient id="planetGrad" cx="35%" cy="35%">
          <stop offset="0%" stop-color="{planet_color}dd"/>
          <stop offset="100%" stop-color="{planet_color}88"/>
        </radialGradient>
        <filter id="glow">
          <feGaussianBlur stdDeviation="2.5" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      <rect width="{SVG_W}" height="{SVG_H}" rx="10" fill="#1e2130"/>
      <!-- stars -->
      <circle cx="20" cy="15" r="1" fill="white" opacity="0.5"/>
      <circle cx="140" cy="25" r="1.2" fill="white" opacity="0.4"/>
      <circle cx="220" cy="10" r="0.8" fill="white" opacity="0.6"/>
      <circle cx="260" cy="40" r="1" fill="white" opacity="0.3"/>
      <circle cx="50" cy="90" r="0.8" fill="white" opacity="0.4"/>
      <!-- Earth -->
      <circle cx="{cx_earth}" cy="{cy}" r="{r_earth}" fill="{earth_grad}" filter="url(#glow)"/>
      <text x="{cx_earth}" y="{cy + r_earth + 14}" text-anchor="middle" fill="#aaa" font-size="11" font-family="sans-serif">Earth</text>
      <!-- Planet -->
      <circle cx="{cx_planet}" cy="{cy}" r="{r_planet:.1f}" fill="{planet_grad}" stroke="{planet_color}" stroke-width="1.5" filter="url(#glow)"/>
      <text x="{cx_planet}" y="{cy + r_planet + 14}" text-anchor="middle" fill="{planet_color}" font-size="11" font-family="sans-serif">{p_rad:.2f}× Earth</text>
    </svg>
    """
    return svg


def render_hz_indicator(teq, insol):
    """
    Habitable Zone indicator
    HZ อยู่ที่ teq ~ 200–320 K หรือ insol ~ 0.3–1.7 (Earth flux)
    """
    # ใช้ insol เป็นหลักถ้ามี, fallback ไป teq
    if insol > 0:
        val = insol
        unit = "Earth flux"
        hz_min, hz_max = 0.3, 1.7
        axis_max = 5.0
        label_val = f"{insol:.2f} S⊕"
    else:
        val = teq
        unit = "K"
        hz_min, hz_max = 200, 320
        axis_max = 1500
        label_val = f"{teq:.0f} K"

    in_hz = hz_min <= val <= hz_max
    too_hot = val > hz_max
    too_cold = val < hz_min

    if in_hz:
        hz_status = "🌿 In Habitable Zone"
        hz_color = "#00CC96"
        hz_desc = "อุณหภูมิเหมาะสมต่อการมีน้ำเหลว"
    elif too_hot:
        hz_status = "🔥 Too Hot"
        hz_color = "#EF553B"
        hz_desc = "ร้อนเกินไป — น้ำระเหยหมด"
    else:
        hz_status = "🧊 Too Cold"
        hz_color = "#636EFA"
        hz_desc = "หนาวเกินไป — น้ำแข็งทั้งหมด"

    # คำนวณ position บน bar (0–100%)
    pct = min(max(val / axis_max, 0), 1) * 100
    hz_start_pct = (hz_min / axis_max) * 100
    hz_end_pct = min((hz_max / axis_max) * 100, 100)

    return hz_status, hz_color, hz_desc, label_val, pct, hz_start_pct, hz_end_pct, in_hz


def show_ai_reasoning(data, prob, threshold):
    margin = min(0.15, threshold * 0.5)
    if prob >= threshold + margin:
        is_verdict = 'CONFIRMED'
    elif prob >= threshold - margin:
        is_verdict = 'CANDIDATE'
    else:
        is_verdict = 'FALSE POSITIVE'

    obj_name = data['kepoi_name'].iloc[0] if 'kepoi_name' in data.columns else "Manual Input Object"
    is_nasa = "st_teff" in data.columns or (data.get('is_nasa', pd.Series([False])).iloc[0] if 'is_nasa' in data.columns else False)
    p_rad   = float(data['koi_prad'].iloc[0])
    teq     = float(data['koi_teq'].iloc[0])
    insol   = float(data['koi_insol'].iloc[0]) if 'koi_insol' in data.columns else 0.0
    snr     = float(data['koi_model_snr'].iloc[0])
    fp_sum  = int(data['koi_fpflag_nt'].iloc[0] + data['koi_fpflag_ss'].iloc[0] +
                  data['koi_fpflag_co'].iloc[0] + data['koi_fpflag_ec'].iloc[0])

    st.markdown("---")
    source_tag = "📡 NASA Data" if is_nasa else "📂 Local Database"
    st.markdown(f"<h2 style='text-align: center; color: #00CC96;'>🪐 Currently Analyzing: {obj_name}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #808495;'>Source: <b>{source_tag}</b></p>", unsafe_allow_html=True)

    with st.expander("View Raw Source Data"):
        st.json(data.to_dict(orient='records')[0])

    res_col1, res_col2 = st.columns([1, 2])

    # ─────────────────────────────────────────
    # คอลัมน์ซ้าย: Verdict + Physical Scale + HZ
    # ─────────────────────────────────────────
    with res_col1:
        st.markdown("### 🎯 AI Verdict")

        verdict_cfg = {
            'CONFIRMED':     ('#00CC96', '#009970', 'rgba(0,204,150,0.2)',  'ดาวเคราะห์',       'CONFIRMED EXOPLANET'),
            'CANDIDATE':     ('#FFC107', '#FF9800', 'rgba(255,193,7,0.2)',  'รอการยืนยัน',      'CANDIDATE'),
            'FALSE POSITIVE':('#EF553B', '#C23B22', 'rgba(239,85,59,0.2)', 'ไม่ใช่ดาวเคราะห์', 'FALSE POSITIVE'),
        }
        vc_color1, vc_color2, vc_shadow, vc_thai, vc_eng = verdict_cfg[is_verdict]
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {vc_color1}, {vc_color2}); padding: 25px 15px; border-radius: 15px;
                    margin-bottom: 20px; text-align: center; box-shadow: 0 8px 16px {vc_shadow};">
            <h2 style="color:white;margin:0;font-size:32px;font-weight:900;letter-spacing:1px;
                       text-shadow:1px 1px 2px rgba(0,0,0,0.2);">{vc_thai}</h2>
            <div style="color:white;font-size:16px;opacity:0.9;margin-bottom:10px;font-weight:500;">
                ({vc_eng})</div>
            <div style="background:rgba(0,0,0,0.2);border-radius:10px;padding:10px;display:inline-block;">
                <div style="color:white;font-size:36px;font-weight:900;line-height:1;">{prob:.2%}</div>
                <div style="color:white;font-size:13px;opacity:0.9;margin-top:5px;">AI Confidence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.metric("System Threshold", f"{threshold:.3f}")

        # ── [1] Physical Scale SVG ──────────────────
        st.markdown("#### 📏 Physical Scale")
        st.markdown(render_planet_scale_svg(p_rad, is_verdict), unsafe_allow_html=True)

        # ── [3] Habitable Zone Indicator ────────────
        st.markdown("#### 🌍 Habitable Zone")
        hz_status, hz_color, hz_desc, label_val, pct, hz_s, hz_e, in_hz = render_hz_indicator(teq, insol)

        st.markdown(f"""
        <div style="background:#1e2130; padding:12px 15px; border-radius:10px; margin-top:6px;">
          <div style="font-size:15px; font-weight:700; color:{hz_color}; margin-bottom:6px;">{hz_status}</div>
          <div style="font-size:12px; color:#aaa; margin-bottom:10px;">{hz_desc} &nbsp;|&nbsp; {label_val}</div>
          <!-- bar -->
          <div style="position:relative; height:14px; background:#2d3147; border-radius:7px; overflow:hidden;">
            <!-- HZ zone highlight -->
            <div style="position:absolute; left:{hz_s:.1f}%; width:{hz_e-hz_s:.1f}%; height:100%;
                        background:rgba(0,204,150,0.35); border-radius:3px;"></div>
            <!-- current value marker -->
            <div style="position:absolute; left:calc({pct:.1f}% - 4px); top:1px;
                        width:8px; height:12px; background:{hz_color}; border-radius:3px;
                        box-shadow: 0 0 6px {hz_color};"></div>
          </div>
          <div style="display:flex; justify-content:space-between; font-size:10px; color:#666; margin-top:4px;">
            <span>Cold</span><span style="color:#00CC96">HZ</span><span>Hot</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # คอลัมน์ขวา: Scorecard + Feature Importance + Gauges
    # ─────────────────────────────────────────
    with res_col2:
        st.markdown("### 🧠 Why did AI decide this?")

        # ── Scorecard ──────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Signal Quality**")
        if snr > 20:   c1.markdown("✅ Excellent")
        elif snr > 10: c1.markdown("🟡 Moderate")
        else:          c1.markdown("❌ Weak Signal")

        c2.markdown("**Planet Size**")
        if p_rad < 0.5:     c2.markdown("🔵 Sub-Earth")
        elif p_rad < 1.25:  c2.markdown("✅ Earth-like")
        elif p_rad < 2.0:   c2.markdown("🟢 Super-Earth")
        elif p_rad < 4.0:   c2.markdown("🟢 Mini-Neptune")
        elif p_rad < 6.0:   c2.markdown("🟡 Neptune-like")
        elif p_rad < 10.0:  c2.markdown("🟡 Saturn-like")
        elif p_rad < 15.0:  c2.markdown("🟠 Jupiter-like")
        else:               c2.markdown("🔴 Super-Jupiter")

        c3.markdown("**Signal Purity**")
        if fp_sum == 0: c3.markdown("✅ No Flags")
        else:           c3.markdown(f"🚩 {fp_sum} Flags")

        # ── [2] Feature Importance Bar ─────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**📊 Key Factors (Feature Contribution)**")
        feat_scores = get_feature_scores(data, model)
        sorted_feats = sorted(feat_scores.items(), key=lambda x: x[1], reverse=True)

        bar_colors = ['#00CC96','#1E90FF','#FFC107','#AB63FA','#FF6692','#B6E880','#FF97FF']
        bars_html = ""
        for i, (name, score) in enumerate(sorted_feats):
            pct_bar = score * 100
            color = bar_colors[i % len(bar_colors)]
            bars_html += f"""
            <div style="margin-bottom:6px;">
              <div style="display:flex; justify-content:space-between; font-size:12px; color:#ccc; margin-bottom:2px;">
                <span>{name}</span><span style="color:{color}; font-weight:600;">{pct_bar:.1f}%</span>
              </div>
              <div style="background:#2d3147; border-radius:4px; height:8px; overflow:hidden;">
                <div style="width:{pct_bar:.1f}%; height:100%; background:{color};
                            border-radius:4px; transition:width 0.3s;"></div>
              </div>
            </div>"""

        st.markdown(f"""
        <div style="background:#1e2130; padding:14px; border-radius:10px;">{bars_html}</div>
        """, unsafe_allow_html=True)

        # ── Gauges ─────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        gauge_col1, gauge_col2 = st.columns(2)

        fig_rad = go.Figure(go.Indicator(
            mode="gauge+number", value=p_rad,
            title={'text': "Radius (Earth=1)"},
            gauge={
                'axis': {'range': [0, 20]},
                'bar': {'color': "#1E90FF"},
                'steps': [
                    {'range': [0, 2],  'color': "rgba(0,204,150,0.3)"},
                    {'range': [2, 6],  'color': "rgba(255,165,0,0.3)"},
                    {'range': [6, 20], 'color': "rgba(239,85,59,0.3)"},
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 2.0}
            }
        ))
        fig_rad.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
        gauge_col1.plotly_chart(fig_rad, use_container_width=True)

        fig_temp = go.Figure(go.Indicator(
            mode="gauge+number", value=teq,
            title={'text': "Eq. Temp (K)"},
            gauge={
                'axis': {'range': [0, 2000]},
                'bar': {'color': "#FF6347"},
                'steps': [
                    {'range': [200, 320], 'color': "rgba(0,204,150,0.35)"},
                    {'range': [320, 800], 'color': "rgba(255,165,0,0.3)"},
                ],
            }
        ))
        fig_temp.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
        gauge_col2.plotly_chart(fig_temp, use_container_width=True)

@st.dialog("🌍 เจาะลึกข้อมูลดาว (Planet Inspection)", width="large")
def show_planet_popup(data, prob, threshold):
    show_ai_reasoning(data, prob, threshold)

# 8. Load Model

model_path = os.path.join(PROJECT_ROOT, 'models', 'kepler_finetuned.pkl') 

if os.path.exists(model_path):
    artifacts = joblib.load(model_path)
    model = artifacts['model']
    best_thresh = artifacts['best_threshold']

    tab1, tab2 = st.tabs(["📂 Batch Processing Dashboard", "🖥️ Manual Analysis"])

    # ==========================================
    # TAB 1: Batch Processing
    # ==========================================
    with tab1:
        st.markdown("### 📊 Automated Mass Vetting Dashboard")
        
        uploaded_file = st.file_uploader("Upload Telemetry File (.csv)", type=['csv'])

        if uploaded_file:
            current_source = uploaded_file.name
            if st.session_state.get('last_batch_source') != current_source:
                st.session_state['batch_results'] = None
                st.session_state['last_batch_source'] = current_source
                # [FIX] clear selected_inspect เมื่อเปลี่ยน source ด้วย
                st.session_state.pop('selected_inspect', None)

            try:
                batch_df = pd.read_csv(uploaded_file, comment='#')
                st.session_state['batch_df_raw'] = batch_df
                
                active_df = st.session_state.get('batch_df_raw', pd.DataFrame())
                if not active_df.empty:
                    st.write(f"📡 วัตถุอวกาศในเซสชันนี้: **{len(active_df):,}** รายการ")
                    
                    if st.button("🚀 Start AI Discovery Engine", type="primary", use_container_width=True):
                        with st.spinner('AI กำลังวิเคราะห์...'):
                            prepared_batch = prepare_input(active_df)
                            probs = model.predict_proba(prepared_batch)[:, 1]
                            active_df = active_df.copy()
                            active_df['AI_Confidence'] = probs
                            
                            margin = min(0.15, best_thresh * 0.5)
                            cond_list = [
                                probs >= (best_thresh + margin),
                                probs >= (best_thresh - margin)
                            ]
                            choice_list = ['CONFIRMED', 'CANDIDATE']
                            active_df['AI_Prediction'] = np.select(cond_list, choice_list, default='FALSE POSITIVE')
                            st.session_state['batch_results'] = active_df

                    if st.session_state.get('batch_results') is not None:
                        res_df = st.session_state['batch_results']
                        
                        st.markdown("---")
                        st.subheader("🎯 Analysis Dashboard")
                        
                        m1, m2, m3, m4 = st.columns(4)
                        total = len(res_df)
                        conf = (res_df['AI_Prediction'] == 'CONFIRMED').sum()
                        m1.metric("Total Objects", f"{total:,}")
                        m2.metric("✅ Confirmed", f"{conf:,}", delta=f"{(conf/total)*100:.1f}%")
                        m3.metric("❌ False Positives", f"{total-conf:,}")
                        m4.metric("Avg Confidence", f"{res_df['AI_Confidence'].mean():.2%}")

                        c_col1, c_col2 = st.columns(2)
                        with c_col1:
                            fig_pie = px.pie(res_df, names='AI_Prediction', hole=0.4, title='Classification Ratio',
                                            color='AI_Prediction', color_discrete_map={'CONFIRMED':'#00CC96', 'CANDIDATE':'#FFC107', 'FALSE POSITIVE':'#EF553B'})
                            st.plotly_chart(fig_pie, width="stretch")
                        with c_col2:
                            if 'koi_period' in res_df.columns and 'koi_prad' in res_df.columns:
                                has_name = 'kepoi_name' in res_df.columns
                                plot_df = res_df.copy()
                                if not has_name:
                                    plot_df['_idx'] = plot_df.index.astype(str)
                                    
                                fig_scatter = px.scatter(plot_df, x='koi_period', y='koi_prad', color='AI_Prediction',
                                                        log_x=True, log_y=True, title='Radius vs Period',
                                                        custom_data=['kepoi_name'] if has_name else ['_idx'],
                                                        hover_data=['kepoi_name'] if has_name else None,
                                                        color_discrete_map={'CONFIRMED':'#00CC96', 'CANDIDATE':'#FFC107', 'FALSE POSITIVE':'#EF553B'})
                                
                                fig_scatter.update_layout(clickmode='event+select', hoverlabel=dict(bgcolor="rgba(0,0,0,0.8)", font_size=14))
                                event = st.plotly_chart(fig_scatter, width="stretch", on_select="rerun", selection_mode="points")
                                
                                # [FIX] ป้องกัน rerun loop โดยเช็คว่า value เปลี่ยนจริงก่อน set
                                if event and event.selection and len(event.selection.points) == 1:
                                    pt = event.selection.points[0]
                                    if 'customdata' in pt:
                                        clicked_val = pt['customdata'][0]
                                        if st.session_state.get('selected_inspect') != clicked_val:
                                            st.session_state['selected_inspect'] = clicked_val
                                            st.session_state.pop('_last_scatter_trigger', None)
                                            st.rerun()
                            else:
                                fig_hist = px.histogram(res_df, x='AI_Confidence', title='Confidence Distribution')
                                st.plotly_chart(fig_hist, width="stretch")

                        st.markdown("---")
                        ins_col1, ins_col2 = st.columns([1, 1])
                        with ins_col1:
                            st.subheader("📋 Analysis Results")
                        with ins_col2:
                            inspect_list = res_df[res_df['AI_Prediction'].isin(['CONFIRMED', 'CANDIDATE'])].sort_values('AI_Confidence', ascending=False)
                            if len(inspect_list) == 0: inspect_list = res_df
                            
                            default_name = st.session_state.get('selected_inspect')
                            names = inspect_list['kepoi_name'].tolist() if 'kepoi_name' in res_df.columns else inspect_list.index.tolist()
                            try: def_idx = names.index(default_name)
                            except: def_idx = 0
                            
                            def format_option(target_name):
                                if 'kepoi_name' in res_df.columns:
                                    row = res_df[res_df['kepoi_name'] == target_name].iloc[0]
                                else:
                                    row = res_df.loc[target_name]
                                icon = "🪐" if row['AI_Prediction'] == 'CONFIRMED' else ("⏳" if row['AI_Prediction'] == 'CANDIDATE' else "❌")
                                return f"{icon} {target_name} ({row['AI_Confidence']:.1%})"

                            sc1, sc2 = st.columns([3, 1])
                            with sc1:
                                selected_inspect = st.selectbox("Select target to inspect:", names, index=def_idx, key="batch_inspect", format_func=format_option)
                            with sc2:
                                st.markdown("<br>", unsafe_allow_html=True)
                                trigger_popup = st.button("📋 เปิด Popup", use_container_width=True)

                        if selected_inspect:
                            if 'kepoi_name' in res_df.columns:
                                selected_data = res_df[res_df['kepoi_name'] == selected_inspect].head(1)
                            else:
                                selected_data = res_df.loc[[selected_inspect]].head(1)
                            
                            if trigger_popup:
                                show_planet_popup(selected_data, selected_data['AI_Confidence'].iloc[0], best_thresh)
                            else:
                                show_ai_reasoning(selected_data, selected_data['AI_Confidence'].iloc[0], best_thresh)

                        with st.expander("📝 View Raw Data Table"):
                            st.dataframe(res_df[['kepoi_name', 'AI_Confidence', 'AI_Prediction']] if 'kepoi_name' in res_df.columns else res_df, use_container_width=True)

                        buffer = io.BytesIO()
                        try:
                            from xlsxwriter.utility import xl_col_to_name
                            pred_col_idx = res_df.columns.get_loc('AI_Prediction')
                            pred_col_letter = xl_col_to_name(pred_col_idx)
                            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                res_df.to_excel(writer, index=False, sheet_name='AI_Results')
                                workbook = writer.book
                                worksheet = writer.sheets['AI_Results']
                                green = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
                                yellow = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C6500'})
                                red = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
                                worksheet.conditional_format(f'A2:ZZ{len(res_df)+1}', {'type': 'formula', 'criteria': f'=${pred_col_letter}2="CONFIRMED"', 'format': green})
                                worksheet.conditional_format(f'A2:ZZ{len(res_df)+1}', {'type': 'formula', 'criteria': f'=${pred_col_letter}2="CANDIDATE"', 'format': yellow})
                                worksheet.conditional_format(f'A2:ZZ{len(res_df)+1}', {'type': 'formula', 'criteria': f'=${pred_col_letter}2="FALSE POSITIVE"', 'format': red})
                            st.download_button("📥 Download Excel Report", buffer, "Kepler_AI_Report.xlsx", "application/vnd.ms-excel", use_container_width=True)
                        except Exception as e:
                            st.warning(f"⚠️ Excel Export unavailable: {e}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


    # ==========================================
    # TAB 2: Manual Analysis
    # ==========================================
    with tab2:
        col_preset1, col_preset2, col_preset3 = st.columns([1, 2, 1])
        with col_preset2:
            preset_options = {
                "Manual Input": [0.0]*18,
                "Kepler-22b (Confirmed)": [289.86, 7.40, 492.0, 2.38, 35.8, 262.0, 0.98, 5518.0, 4.44, 1.11, 15.3, 0, 0, 0, 0, 0.007, 0.05, 1.0],
                "Eclipsing Binary (False Positive)": [1.50, 2.00, 50000.0, 15.00, 1500.0, 4500.0, 1.20, 6000.0, 4.00, 10000.0, 10.5, 0, 1, 1, 0, 0.15, 0.02, 0.0]
            }
            selected_preset = st.selectbox("📂 Load Presets:", list(preset_options.keys()))
            default_vals = preset_options[selected_preset]

        st.markdown("### 🎛️ Telemetry Parameters")
        with st.form("manual_analysis_form", border=True):
            c1, c2, c3, c4 = st.columns(4)
            koi_period = c1.number_input("Period (Days)", value=default_vals[0], help="ระยะเวลาที่ใช้ในการโคจรรอบดาวฤกษ์ 1 รอบ")
            koi_duration = c2.number_input("Duration (Hours)", value=default_vals[1], help="ระยะเวลาที่ดาวเคราะห์ใช้ในการเดินผ่านหน้าดาวฤกษ์")
            koi_depth = c3.number_input("Depth (PPM)", value=default_vals[2], help="ความสว่างของดาวฤกษ์ที่ลดลงเมื่อมีวัตถุมาบัง (หน่วย: ส่วนในล้าน)")
            koi_prad = c4.number_input("Planet Radius", value=default_vals[3], help="ขนาดรัศมีเมื่อเทียบกับโลก (1 = ขนาดเท่าโลก)")

            c5, c6, c7, c8 = st.columns(4)
            koi_model_snr = c5.number_input("SNR (Signal-to-Noise)", value=default_vals[4], help="ความชัดของสัญญาณเทียบกับสัญญาณรบกวน")
            koi_teq = c6.number_input("Temp (K)", value=default_vals[5], help="อุณหภูมิพื้นผิวโดยประมาณ (หน่วย: เคลวิน)")
            koi_srad = c7.number_input("Stellar Radius", value=default_vals[6], help="ขนาดของดาวฤกษ์แม่เทียบกับดวงอาทิตย์")
            koi_steff = c8.number_input("Stellar Temp", value=default_vals[7], help="อุณหภูมิของดาวฤกษ์แม่")
            
            c9, c10, c11 = st.columns(3)
            koi_slogg = c9.number_input("Gravity log(g)", value=default_vals[8])
            koi_insol = c10.number_input("Insolation Flux", value=default_vals[9])
            koi_kepmag = c11.number_input("Kepler Mag", value=default_vals[10])
            
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🚀 Run Advanced Classification", type="primary", use_container_width=True)

        koi_score = float(default_vals[17])
        koi_fpflag_nt = bool(default_vals[11])
        koi_fpflag_ss = bool(default_vals[12])
        koi_fpflag_co = bool(default_vals[13])
        koi_fpflag_ec = bool(default_vals[14])
        koi_ror = default_vals[15]
        koi_impact = default_vals[16]
        koi_srho = 1.0
        koi_sma = 0.1
        koi_incl = 89.0
        koi_dor = 10.0

        if submitted:
            raw_dict = {
                'koi_period': koi_period, 'koi_duration': koi_duration, 'koi_depth': koi_depth,
                'koi_prad': koi_prad, 'koi_model_snr': koi_model_snr, 'koi_teq': koi_teq,
                'koi_srad': koi_srad, 'koi_steff': koi_steff, 'koi_slogg': koi_slogg,
                'koi_insol': koi_insol, 'koi_kepmag': koi_kepmag, 'koi_fpflag_nt': int(koi_fpflag_nt),
                'koi_fpflag_ss': int(koi_fpflag_ss), 'koi_fpflag_co': int(koi_fpflag_co),
                'koi_fpflag_ec': int(koi_fpflag_ec), 'koi_ror': koi_ror, 'koi_impact': koi_impact,
                'koi_srho': koi_srho, 'koi_sma': koi_sma, 'koi_incl': koi_incl, 'koi_dor': koi_dor,
                'koi_score': koi_score
            }
            raw_df = pd.DataFrame([raw_dict])
            prepared_data = prepare_input(raw_df)
            
            with st.spinner('Analyzing...'):
                prob = model.predict_proba(prepared_data)[0][1]
                show_ai_reasoning(raw_df, prob, best_thresh)




else:
    st.error("❌ Model not found. Please run training first.")