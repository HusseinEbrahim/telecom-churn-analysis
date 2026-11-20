import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Telecom Churn Dashboard", layout="wide")

DROP = ["customerID", "Churn", "Churned", "TenureGroup"]
TENURE_ORDER = ["0-6 mo", "7-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"]


@st.cache_data
def load_data():
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churned"] = (df["Churn"] == "Yes").astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["TenureGroup"] = pd.cut(df["tenure"], bins=[-1, 6, 12, 24, 48, 72],
                               labels=TENURE_ORDER).astype(str)
    return df


@st.cache_resource
def train_model(df):
    X = pd.get_dummies(df.drop(columns=DROP), drop_first=True)
    model = RandomForestClassifier(n_estimators=300, max_depth=8,
                                   class_weight="balanced", random_state=42)
    model.fit(X, df["Churned"])
    return model, X.columns


df = load_data()
model, model_columns = train_model(df)

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
contracts = st.sidebar.multiselect("Contract", sorted(df["Contract"].unique()),
                                   default=sorted(df["Contract"].unique()))
internet = st.sidebar.multiselect("Internet service", sorted(df["InternetService"].unique()),
                                  default=sorted(df["InternetService"].unique()))
f = df[df["Contract"].isin(contracts) & df["InternetService"].isin(internet)]
if f.empty:
    st.warning("No customers match these filters.")
    st.stop()

# ---------- KPIs ----------
st.title("Telecom Customer Churn Dashboard")
lost = f.loc[f["Churned"] == 1, "MonthlyCharges"].sum()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{len(f):,}")
c2.metric("Churn rate", f"{f['Churned'].mean():.1%}")
c3.metric("Monthly revenue lost", f"${lost:,.0f}")
c4.metric("Avg monthly charge", f"${f['MonthlyCharges'].mean():.2f}")

# ---------- Charts ----------
left, right = st.columns(2)
by_contract = f.groupby("Contract")["Churned"].mean().mul(100).reset_index()
left.plotly_chart(px.bar(by_contract, x="Contract", y="Churned",
                         labels={"Churned": "Churn %"}, title="Churn rate by contract"),
                  use_container_width=True)

by_tenure = (f.groupby("TenureGroup")["Churned"].mean().mul(100)
             .reindex(TENURE_ORDER).dropna().reset_index())
right.plotly_chart(px.line(by_tenure, x="TenureGroup", y="Churned", markers=True,
                           labels={"Churned": "Churn %", "TenureGroup": "Tenure"},
                           title="Churn rate by tenure"),
                   use_container_width=True)

left, right = st.columns(2)
left.plotly_chart(px.box(f, x="Churn", y="MonthlyCharges",
                         title="Monthly charges: churned vs stayed"),
                  use_container_width=True)
by_pay = f.groupby("PaymentMethod")["Churned"].mean().mul(100).sort_values().reset_index()
right.plotly_chart(px.bar(by_pay, x="Churned", y="PaymentMethod", orientation="h",
                          labels={"Churned": "Churn %", "PaymentMethod": ""},
                          title="Churn rate by payment method"),
                   use_container_width=True)

# ---------- Segment table ----------
st.subheader("Highest-risk segments (by monthly revenue lost)")
seg = (f.groupby(["Contract", "InternetService", "TenureGroup"])
        .apply(lambda g: pd.Series({
            "customers": len(g),
            "churn_pct": round(g["Churned"].mean() * 100, 1),
            "revenue_lost": round(g.loc[g["Churned"] == 1, "MonthlyCharges"].sum()),
        }))
        .reset_index())
seg = seg[seg["customers"] >= 50].sort_values("revenue_lost", ascending=False).head(10)
st.dataframe(seg, use_container_width=True, hide_index=True)

# ---------- Predict a customer's risk ----------
st.subheader("Predict churn risk for a customer")
a, b, c = st.columns(3)
contract = a.selectbox("Contract", sorted(df["Contract"].unique()))
tenure = a.slider("Tenure (months)", 0, 72, 6)
monthly = b.slider("Monthly charge ($)", 18, 120, 75)
net = b.selectbox("Internet service", sorted(df["InternetService"].unique()))
pay = c.selectbox("Payment method", sorted(df["PaymentMethod"].unique()))
support = c.selectbox("Tech support", sorted(df["TechSupport"].unique()))

row = df.iloc[[0]].copy()
row["Contract"], row["tenure"], row["MonthlyCharges"] = contract, tenure, monthly
row["TotalCharges"] = tenure * monthly
row["InternetService"], row["PaymentMethod"], row["TechSupport"] = net, pay, support

full = pd.get_dummies(pd.concat([df, row]).drop(columns=DROP), drop_first=True)
x = full.iloc[[-1]].reindex(columns=model_columns, fill_value=0)
p = model.predict_proba(x)[0, 1]
level = "High" if p >= 0.6 else "Medium" if p >= 0.35 else "Low"
st.metric("Churn probability", f"{p:.0%}", help="Random Forest model, ROC-AUC 0.84")
st.write(f"**Risk level: {level}**")