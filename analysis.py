import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')
import os
os.makedirs('charts', exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
users       = pd.read_csv("data/users.csv")
sessions    = pd.read_csv("data/sessions.csv")
chat_topics = pd.read_csv("data/chat_topics.csv")

print(f"Users: {len(users):,}  |  Sessions: {len(sessions):,}")

# ── Build analysis dataset ────────────────────────────────────────────────────
first_sessions = (sessions.sort_values(["user_id","session_date"])
                  .groupby("user_id").first().reset_index())
day1_counts = sessions.groupby("user_id").size().rename("day1_sessions").reset_index()

df = (users
      .merge(first_sessions[["user_id","session_length_minutes","sentiment_score"]], on="user_id", how="left")
      .merge(day1_counts, on="user_id", how="left")
      .merge(chat_topics, on="user_id", how="left"))

df["day1_sessions"]        = df["day1_sessions"].fillna(0).astype(int)
df["two_plus_sessions"]    = df["day1_sessions"] >= 2
df["first_session_over_3"] = df["session_length_minutes"] > 3
df["length_bucket"]        = pd.cut(df["session_length_minutes"],
                                     bins=[0,2,5,10,200],
                                     labels=["<2 min","2-5 min","5-10 min","10+ min"])

# ── Question 1: Basic retention ───────────────────────────────────────────────
print("\n── Basic Metrics ──")
print(f"Day 1 retention:  {users['returned_day_1'].mean():.1%}")
print(f"Day 7 retention:  {users['returned_day_7'].mean():.1%}")
print(f"Paid conversion:  {users['converted_paid'].mean():.1%}")
print("\nDay 7 by channel:")
print(users.groupby("acquisition_channel")["returned_day_7"].mean().sort_values(ascending=False).round(3))
print("\nDay 7 by voice usage:")
print(users.groupby("used_voice_first_day")["returned_day_7"].mean().round(3))

# ── Question 2: Activation behavior ──────────────────────────────────────────
print("\n── Activation Behavior ──")
segments = {
    "Used voice Day 1":        df["used_voice_first_day"] == 1,
    "Text-only (no voice)":    df["used_voice_first_day"] == 0,
    "2+ sessions Day 1":       df["two_plus_sessions"] == True,
    "1 session Day 1":         df["two_plus_sessions"] == False,
    "Returned Day 1":          df["returned_day_1"] == 1,
    "Did not return Day 1":    df["returned_day_1"] == 0,
    "Session > 3 min":         df["first_session_over_3"] == True,
    "Session <= 3 min":        df["first_session_over_3"] == False,
}
for label, mask in segments.items():
    print(f"  {label:<28} {df.loc[mask,'returned_day_7'].mean():.1%}  (n={mask.sum():,})")

print("\nDay 7 by session length bucket:")
print(df.groupby("length_bucket")["returned_day_7"].agg(["mean","count"]).round(3))

# ── Question 3: Topic analysis ────────────────────────────────────────────────
print("\n── Topic Analysis ──")
print("Top topics by volume:")
print(df["first_topic"].value_counts().head(8))
print("\nDay 7 retention by topic:")
print(df.groupby("first_topic")["returned_day_7"].agg(["mean","count"]).sort_values("mean",ascending=False).round(3))
print(f"\nCrisis flags: {df['contains_crisis_language'].sum()} users")

# ── Question 4: Predictive model ──────────────────────────────────────────────
print("\n── Predictive Model ──")
features = ["used_voice_first_day","used_text_first_day","day1_sessions",
            "session_length_minutes","sentiment_score",
            "acquisition_channel","age_bucket","country","first_topic","topic_cluster"]
model_df = df[features + ["returned_day_7"]].dropna()
X = pd.get_dummies(model_df[features], drop_first=True)
y = model_df["returned_day_7"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)):.3f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.3f}")
coef_df = (pd.DataFrame({"feature": X.columns, "coefficient": model.coef_[0]})
           .assign(abs=lambda d: d["coefficient"].abs())
           .sort_values("abs", ascending=False))
print("\nTop 5 predictors:")
print(coef_df.head(5)[["feature","coefficient"]].to_string(index=False))

# ── Charts ────────────────────────────────────────────────────────────────────
C_BLUE='#2563a8'; C_GREEN='#1a7f5a'; C_CORAL='#c94040'
C_PURPLE='#6d5bb5'; C_AMBER='#b06c10'; C_GRAY='#6b6b67'; C_DARK='#1a1a18'

def no_spines(ax):
    for s in ax.spines.values(): s.set_visible(False)

# Funnel
fig, ax = plt.subplots(figsize=(8,3.2)); ax.set_facecolor('white'); fig.patch.set_facecolor('white')
for i,(lbl,val,pct,col) in enumerate(zip(['Signed up','Came back Day 1','Came back Day 7','Became paid'],
                                          [7500,3740,3425,833],[100,49.9,45.7,11.1],
                                          [C_BLUE,C_PURPLE,C_GREEN,C_AMBER])):
    ax.barh(i, val/7500, color=col, height=0.55, zorder=3, alpha=0.92)
    ax.text(val/7500+0.01, i, f'{val:,}  ({pct:.1f}%)', va='center', fontsize=10.5, color='#333')
ax.set_yticks(range(4)); ax.set_yticklabels(['Signed up','Came back Day 1','Came back Day 7','Became paid'], fontsize=11, color='#444')
ax.set_xlim(0,1.55); ax.set_xticks([]); no_spines(ax)
ax.set_title('The journey from signup to paid customer', fontsize=13, fontweight='bold', color=C_DARK, loc='left', pad=10)
fig.tight_layout(); fig.savefig('charts/user_funnel.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close()

# Channel
ch = users.groupby('acquisition_channel')['returned_day_7'].mean().sort_values()*100
fig, ax = plt.subplots(figsize=(8,3.2)); ax.set_facecolor('white'); fig.patch.set_facecolor('white')
cols = [C_CORAL if v<45.7 else (C_GREEN if v>46.5 else C_BLUE) for v in ch.values]
ax.barh(range(len(ch)), ch.values, color=cols, height=0.55, zorder=3, alpha=0.9)
ax.axvline(45.7, color='#aaa', linestyle='--', linewidth=1.2, zorder=4)
ax.text(45.8, len(ch)-0.4, 'avg 45.7%', fontsize=9, color='#888', style='italic')
for i,v in enumerate(ch.values): ax.text(v+0.1,i,f'{v:.1f}%',va='center',fontsize=10,color='#333')
ax.set_yticks(range(len(ch))); ax.set_yticklabels(ch.index, fontsize=11, color='#444')
ax.set_xlim(40,52); ax.set_xticks([]); no_spines(ax)
ax.set_title('Where users came from barely matters for retention', fontsize=13, fontweight='bold', color=C_DARK, loc='left', pad=10)
fig.tight_layout(); fig.savefig('charts/day7_retention_by_channel.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close()

# Activation
act_labels=['Text only, no voice','First session <2 min','First session 2-5 min',
            'Used voice on Day 1','Came back on Day 1','First session 5-10 min',
            'First session 10+ min','2+ sessions on Day 1']
act_vals=[29.4,38.4,46.3,55.6,57.0,55.8,57.1,73.9]
act_cols=[C_CORAL,C_CORAL,C_BLUE,C_GREEN,C_GREEN,C_GREEN,C_GREEN,'#0f5c3a']
fig, ax = plt.subplots(figsize=(8,4.2)); ax.set_facecolor('white'); fig.patch.set_facecolor('white')
ax.barh(range(8), act_vals, color=act_cols, height=0.58, zorder=3, alpha=0.92)
ax.axvline(45.7, color='#bbb', linestyle='--', linewidth=1.2, zorder=4)
ax.text(45.9, 7.6, 'baseline 45.7%', fontsize=8.5, color='#999', style='italic')
for i,v in enumerate(act_vals): ax.text(v+0.4,i,f'{v:.1f}%',va='center',fontsize=10.5,color='#333',fontweight='500')
ax.set_yticks(range(8)); ax.set_yticklabels(act_labels, fontsize=10.5, color='#444')
ax.set_xlim(20,85); ax.set_xticks([]); no_spines(ax)
ax.set_title('What users do on Day 1 predicts if they come back', fontsize=13, fontweight='bold', color=C_DARK, loc='left', pad=10)
fig.tight_layout(); fig.savefig('charts/day7_retention_by_activation.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close()

# Session length
fig, ax = plt.subplots(figsize=(8,3.0)); ax.set_facecolor('white'); fig.patch.set_facecolor('white')
bars = ax.bar(range(4),[38.4,46.3,55.8,57.1],color=[C_CORAL,C_AMBER,C_BLUE,C_GREEN],width=0.55,zorder=3,alpha=0.92)
for bar,v in zip(bars,[38.4,46.3,55.8,57.1]):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.5, f'{v:.1f}%', ha='center', fontsize=11.5, color='#333', fontweight='500')
ax.axhline(45.7, color='#bbb', linestyle='--', linewidth=1.2, zorder=4)
ax.set_xticks(range(4)); ax.set_xticklabels(['Under 2 min','2 to 5 min','5 to 10 min','Over 10 min'], fontsize=11, color='#444')
ax.set_ylim(30,65); ax.set_yticks([]); no_spines(ax)
ax.set_title('Longer first conversations = more likely to come back', fontsize=13, fontweight='bold', color=C_DARK, loc='left', pad=10)
fig.tight_layout(); fig.savefig('charts/day7_retention_by_session_length.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close()

# Topic retention
t = df.groupby('first_topic')['returned_day_7'].mean().sort_values()*100
fig, ax = plt.subplots(figsize=(8,3.2)); ax.set_facecolor('white'); fig.patch.set_facecolor('white')
tcols = [C_CORAL if v<44 else (C_GREEN if v>47 else C_BLUE) for v in t.values]
ax.barh(range(len(t)), t.values, color=tcols, height=0.55, zorder=3, alpha=0.9)
ax.axvline(45.7, color='#bbb', linestyle='--', linewidth=1.2, zorder=4)
for i,v in enumerate(t.values): ax.text(v+0.1,i,f'{v:.1f}%',va='center',fontsize=10,color='#333')
ax.set_yticks(range(len(t))); ax.set_yticklabels(t.index, fontsize=11, color='#444')
ax.set_xlim(39,53); ax.set_xticks([]); no_spines(ax)
ax.set_title('First conversation topic has modest impact on retention', fontsize=13, fontweight='bold', color=C_DARK, loc='left', pad=10)
fig.tight_layout(); fig.savefig('charts/day7_retention_by_topic.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close()

# Predictors
fig, ax = plt.subplots(figsize=(8,3.0)); ax.set_facecolor('white'); fig.patch.set_facecolor('white')
ax.barh(range(5),[0.040,0.111,0.137,0.611,1.033],
        color=[C_GRAY,C_GRAY,C_BLUE,C_BLUE,C_GREEN],height=0.45,zorder=3,alpha=0.88)
for i,v in enumerate([0.040,0.111,0.137,0.611,1.033]):
    ax.text(v+0.015,i,f'{v:.3f}',va='center',fontsize=10,color='#333')
ax.set_yticks(range(5))
ax.set_yticklabels(['Session length','Used text Day 1','Positive sentiment','# Day 1 sessions','Used voice Day 1'], fontsize=11, color='#444')
ax.set_xlim(0,1.22); ax.set_xticks([]); no_spines(ax)
ax.set_title('Voice usage is by far the strongest retention predictor', fontsize=13, fontweight='bold', color=C_DARK, loc='left', pad=10)
fig.tight_layout(); fig.savefig('charts/model_top_predictors.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close()

print("\nAll charts saved to charts/")
print("Done.")
