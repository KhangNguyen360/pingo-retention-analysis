# Pingo User Retention & Voice Chat Analytics

**Question:** What early user behaviors are most associated with users coming back after 7 days?

Analysis of 7,500 simulated Pingo users across acquisition channels, session behaviors, and first-conversation topics.

---

## Key Finding

Users who used voice on Day 1 retained at **55.6%** vs **29.4%** for text-only users — a 1.9× lift. Users who had 2+ sessions on Day 1 retained at **73.9%**, the strongest signal in the dataset.

## Files

| File | Description |
|---|---|
| `Pingo_Retention_Report.html` | Full report — open in any browser |
| `analysis.py` | Python analysis script |
| `data/users.csv` | User-level data (7,500 rows) |
| `data/sessions.csv` | Session-level data (27,000+ rows) |
| `data/chat_topics.csv` | First conversation topic data |
| `requirements.txt` | Python dependencies |

## How to Run

```bash
pip install -r requirements.txt
mkdir -p data charts
# place users.csv, sessions.csv, chat_topics.csv in data/
python analysis.py
```

## Results

| Metric | Value |
|---|---|
| Day 1 retention | 49.9% |
| Day 7 retention | 45.7% |
| Paid conversion | 11.1% |
| Voice users Day 7 retention | 55.6% |
| 2+ Day 1 sessions retention | 73.9% |
| Model ROC-AUC | 0.70 |

## Top Recommendations

1. **Redesign onboarding** around one meaningful voice session
2. **Send a second-session nudge** within 6 hours of signup
3. **Build a retention dashboard** segmented by early behaviors

---

*Simulated dataset for product analytics practice. All findings are correlational.*
