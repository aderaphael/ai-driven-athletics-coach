import xml.etree.ElementTree as ET
import pandas as pd
import copy
from datetime import datetime
from geopy.distance import geodesic
from transformers import pipeline as hf_pipeline
import json
import streamlit as st

# -------------------------------
# Phase 1: GPX Data Ingestion
# -------------------------------
def parse_gpx_file(file):
    ns = {
        'default': 'http://www.topografix.com/GPX/1/1',
        'gpxtpx': 'http://www.garmin.com/xmlschemas/TrackPointExtension/v1'
    }
    tree = ET.parse(file)
    root = tree.getroot()
    data = []
    for wpt in root.findall('default:wpt', ns):
        lat = float(wpt.attrib['lat'])
        lon = float(wpt.attrib['lon'])
        ele = float(wpt.find('default:ele', ns).text)
        time = datetime.fromisoformat(
            wpt.find('default:time', ns).text.replace('Z', '+00:00')
        )
        hr_elem = wpt.find('.//gpxtpx:hr', ns)
        hr = int(hr_elem.text) if hr_elem is not None else None
        data.append([time, lat, lon, ele, hr])
    df = pd.DataFrame(data, columns=['time','lat','lon','elevation','heart_rate'])
    df = df.sort_values('time').reset_index(drop=True)
    df['time_diff_sec'] = df['time'].diff().dt.total_seconds().fillna(0)
    return df

# -------------------------------
# Phase 2: Feature Engineering
# -------------------------------
def calculate_features(df):
    distances = [0.0]
    for i in range(1, len(df)):
        prev = (df.loc[i-1,'lat'], df.loc[i-1,'lon'])
        curr = (df.loc[i,'lat'], df.loc[i,'lon'])
        distances.append(geodesic(prev, curr).km)
    df['distance_km'] = distances
    df['pace_sec_per_km'] = df['time_diff_sec'] / df['distance_km'].replace(0,1e-6)
    return {
        'total_weekly_distance_km': round(df['distance_km'].sum(),2),
        'average_pace_sec_per_km': round(df['pace_sec_per_km'].mean(),2),
        'average_heart_rate': round(df['heart_rate'].mean(),2),
        'longest_run_km': round(df['distance_km'].max(),2),
        'run_frequency_days': df['time'].dt.date.nunique()
    }

# -------------------------------
# Phase 3: Rule-Based Plan Generator
# -------------------------------
def classify_fitness_level(distance):
    if distance < 30:
        return 'beginner'
    if distance < 60:
        return 'intermediate'
    return 'advanced'

# **Updated**: accept `runs_per_week` explicitly
def generate_week_plan(features, runs_per_week):
    level = classify_fitness_level(features['total_weekly_distance_km'])
    runs = runs_per_week
    if level == 'beginner':
        return ['Easy Run']*(runs-1) + ['Long Run']
    if level == 'intermediate':
        return ['Tempo Run','Intervals'] + ['Easy Run']*(runs-3) + ['Long Run']
    return ['Threshold','Intervals','Easy Run']*(runs//3) + ['Long Run']

# -------------------------------
# Phase 4: VDOT Table Integration
# -------------------------------
vdot_df = pd.read_csv('vdot_table_sample.csv')

def get_vdot_row(user_vdot):
    return vdot_df.iloc[(vdot_df['VDOT'] - user_vdot).abs().argsort()[:1]].squeeze()

def get_session_pace(session, vdot_row):
    if session == 'Easy Run':
        return vdot_row['Easy_Pace_min_per_km']
    if session in ['Tempo Run','Threshold']:
        return vdot_row['Threshold_Pace']
    if session in ['Intervals','Interval']:
        return vdot_row['Interval_Pace']
    if session == 'Repetition':
        return vdot_row['Repetition_Pace']
    return vdot_row['Easy_Pace_min_per_km'] + 0.3

# -------------------------------
# Phase 5: Multi-Week Plan Builder
# -------------------------------
# **Updated**: pass `runs_per_week` through
def generate_12_week_plan(features, runs_per_week):
    base_dist = features['total_weekly_distance_km']
    base_long = features['longest_run_km']
    progression = [1.0,1.05,1.1,1.15,1.2,1.25,1.3,1.35,0.9,0.8,0.7,0.5]
    plan = []
    for i in range(12):
        total_km = round(base_dist * progression[i],1)
        long_km  = round(base_long + min(i//2,4),1)
        sessions = generate_week_plan(features, runs_per_week)
        plan.append({
            'week': i+1,
            'total_target_km': total_km,
            'long_run_km': long_km,
            'sessions': sessions
        })
    return plan

# -------------------------------
# Phase 6: Feedback & Adaptation
# -------------------------------
classifier = hf_pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=-1
)
ZS_LABELS = ["fatigue", "too_easy", "missed_or_injured"]

def feedback_to_structured_json(feedback_text):
    text = feedback_text.lower()
    if "injur" in text:
        return {"fatigue": False, "increase_difficulty": False, "adjust_schedule": True}
    res = classifier(
        sequences=feedback_text,
        candidate_labels=ZS_LABELS,
        multi_label=True
    )
    labels, scores = res["labels"], res["scores"]
    mapping = {label: score for label, score in zip(labels, scores)}
    return {
        "fatigue":           mapping.get("fatigue", 0) > 0.5,
        "increase_difficulty": mapping.get("too_easy", 0) > 0.5,
        "adjust_schedule":     mapping.get("missed_or_injured", 0) > 0.5
    }

def adapt_week_plan(week, fb):
    w = copy.deepcopy(week)
    w['note'] = "No change"
    if fb.get('fatigue'):
        w['sessions'] = ["Easy Run" if s!="Long Run" else s for s in w['sessions']]
        w['note'] = "Reduced intensity"
    elif fb.get('increase_difficulty'):
        new, cnt = [], 0
        for s in w['sessions']:
            if s=="Easy Run" and cnt<2:
                new.append("Tempo Run"); cnt+=1
            else:
                new.append(s)
        w['sessions'] = new; w['note']="Increased challenge"
    elif fb.get('adjust_schedule'):
        if len(w['sessions'])>3:
            w['sessions'] = w['sessions'][:-1]
        w['note']="Shortened week"
    return w

# -------------------------------
# Phase 7: Streamlit UI
# -------------------------------
st.title("AI‑Driven Athletics Coach Prototype")

# 1) GPX upload
uploaded = st.file_uploader("Upload your GPX file", type="gpx")
if not uploaded:
    st.info("Please upload a GPX file to get started.")
    st.stop()

# 2) Parse & features
df    = parse_gpx_file(uploaded)
feats = calculate_features(df)
st.subheader("Extracted Features")
st.json(feats)

# **New** Runs‐per‐week slider
runs_per_week = st.slider(
    "How many sessions per week?", 1, 7, 3
)

# 3) Build multi‐week plan with that frequency
plan     = generate_12_week_plan(feats, runs_per_week)
vdot_row = get_vdot_row(feats.get('vdot', 45))
for wk in plan:
    wk['sessions'] = [
        {'type': s,
         'pace': round(get_session_pace(s, vdot_row), 2)}
        for s in wk['sessions']
    ]

# 4) Week selector
current_week = st.slider("Select week to view", 1, 12, 1)
wk_data = plan[current_week-1]

st.subheader(f"Week {current_week} Plan ({runs_per_week} sessions)")
st.table(pd.DataFrame(wk_data['sessions']))

# 5) Feedback & adaptation
fb_text = st.text_input(f"Feedback on Week {current_week} (e.g. 'I felt knackered'):")
if st.button(f"Adapt Week {current_week}"):
    fb      = feedback_to_structured_json(fb_text)
    st.write("Structured feedback:", fb)
    adapted = adapt_week_plan(wk_data, fb)
    st.subheader(f"Adapted Week {current_week}")
    st.table(pd.DataFrame(adapted['sessions']))
    st.write("Note:", adapted['note'])
