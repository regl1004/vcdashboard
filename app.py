from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Service & Advice Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "synthetic_vom_survey_csvs"
LOW_VOLUME_THRESHOLD = 5
ORG_LABEL = "Organization"
THEME_MAP = {
    "Wait time": ["wait", "busy"],
    "Clarity / next steps": ["clear", "detail", "explain", "follow-up", "follow up"],
    "Listening / empathy": ["listen", "heard", "understand", "warm", "looked after"],
    "Tailored advice": ["tailored", "goals", "specific", "general", "situation"],
    "Ease / effort": ["easy", "repeat", "frustrating", "request was completed"],
    "Value-add": ["plan", "useful", "information", "tools", "community", "appointment"],
}
ROLE_DESCRIPTIONS = {
    "Frontline Coaching": "Signal to action for individual coaching decisions and urgent interventions.",
    "Branch Performance": "Benchmark, prioritize, and monitor your branch across service and advice.",
    "Regional Leadership": "Identify which branches need support and where regional attention should go.",
    "MX Integration": "Understand experience drivers, broad performance themes, and organizational gaps.",
}
SATISFACTION_SCALE = {
    "Very dissatisfied": 1,
    "Dissatisfied": 2,
    "Neither satisfied nor dissatisfied": 3,
    "Satisfied": 4,
    "Very satisfied": 5,
}
WAIT_TIME_SCALE = {
    "Strongly disagree": 1,
    "Somewhat disagree": 2,
    "Neither agree nor disagree": 3,
    "Somewhat agree": 4,
    "Strongly agree": 5,
}
BEHAVIOUR_LABELS = {
    "They contacted me in advance to prepare for the appointment.": "Prepared in advance",
    "They listened carefully and confirmed their understanding of my needs.": "Listened to needs",
    "They tailored their advice to my financial goals and situation.": "Tailored advice",
    "They addressed what I came in for, or clearly explained next steps.": "Clear next steps",
    "They were knowledgeable and clear in this interaction.": "Knowledgeable",
    "They made me feel recognized and genuinely cared for.": "Recognized and cared for",
    "They made it easy for me to get what I needed.": "Easy to get help",
    "They acknowledged me when I arrived (whether at the wicket or upon entry).": "Acknowledged arrival",
    "They listened carefully and confirmed their understanding of my request.": "Listened to request",
    "They either addressed my request during this visit, or clearly explained the next steps.": "Resolved or explained next steps",
    "They were patient, respectful, and had a positive attitude.": "Respectful attitude",
}


@st.cache_data(show_spinner=False)
def load_data() -> dict[str, pd.DataFrame]:
    survey = pd.read_csv(DATA_DIR / "synthetic_survey_responses.csv", encoding="utf-8-sig")
    behaviour = pd.read_csv(
        DATA_DIR / "synthetic_behaviour_statement_responses.csv", encoding="utf-8-sig"
    )
    value_add = pd.read_csv(DATA_DIR / "value_add_response_outcomes_long.csv", encoding="utf-8-sig")
    metadata = pd.read_csv(DATA_DIR / "metadata.csv", encoding="utf-8-sig")

    for frame in (survey, behaviour):
        frame["Survey date"] = pd.to_datetime(frame["Survey date"])

    survey["Satisfaction numeric"] = survey["Satisfaction rating"].map(SATISFACTION_SCALE)
    survey["Wait time score"] = survey["Wait time reasonableness"].map(WAIT_TIME_SCALE)
    survey["Month"] = survey["Survey date"].dt.to_period("M").astype(str)
    behaviour["Month"] = behaviour["Survey date"].dt.to_period("M").astype(str)

    responses = (
        survey.sort_values("Survey date")
        .drop_duplicates(subset=["Response ID"])
        .reset_index(drop=True)
        .copy()
    )

    responses["Negative experience flag"] = (
        (responses["NPS group"] == "Detractor")
        | (responses["Satisfaction group"] == "Dissatisfied")
        | (responses["Follow-up permission flag"] == "Yes")
    )
    responses["Wait time issue flag"] = responses["Wait time score"].fillna(5) <= 2

    behaviour_summary = (
        behaviour.groupby(
            ["Interaction type", "Region", "Branch", "Staff member", "Staff role", "Behaviour statement"],
            dropna=False,
        )
        .agg(
            avg_score=("Behaviour score", "mean"),
            responses=("Response ID", "nunique"),
            agree_rate=("Behaviour agreement group", lambda s: (s == "Agree").mean()),
            detractor_rate=("NPS group", lambda s: (s == "Detractor").mean()),
        )
        .reset_index()
    )
    behaviour_summary["Behaviour short"] = behaviour_summary["Behaviour statement"].map(
        lambda s: BEHAVIOUR_LABELS.get(s, s)
    )

    return {
        "survey": survey,
        "responses": responses,
        "behaviour": behaviour,
        "behaviour_summary": behaviour_summary,
        "value_add": value_add,
        "metadata": metadata,
    }


def apply_filters(
    df: pd.DataFrame,
    *,
    regions: Iterable[str],
    branches: Iterable[str],
    staff_members: Iterable[str],
    staff_roles: Iterable[str],
    interaction_types: Iterable[str],
    appointment_types: Iterable[str],
) -> pd.DataFrame:
    filtered = df.copy()
    mapping = {
        "Region": list(regions),
        "Branch": list(branches),
        "Staff member": list(staff_members),
        "Staff role": list(staff_roles),
        "Interaction type": list(interaction_types),
    }
    for column, values in mapping.items():
        if values and "All" not in values and column in filtered.columns:
            filtered = filtered[filtered[column].isin(values)]

    if "Appointment type" in filtered.columns and appointment_types and "All" not in appointment_types:
        values = [v for v in appointment_types if v != "Service / Not applicable"]
        include_service = "Service / Not applicable" in appointment_types
        mask = filtered["Appointment type"].isin(values)
        if include_service:
            mask = mask | filtered["Appointment type"].isna()
        filtered = filtered[mask]

    return filtered


def metric_card(column, label: str, value: str, delta: str | None = None, help_text: str | None = None) -> None:
    with column:
        if help_text:
            st.caption(help_text)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-delta">{delta or ""}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.1f}%"


def number(value: float | int | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"


def summarize_scores(responses: pd.DataFrame) -> dict[str, float]:
    if responses.empty:
        return {
            "responses": 0,
            "nps": np.nan,
            "avg_nps": np.nan,
            "sat": np.nan,
            "value_add": np.nan,
            "follow_up": np.nan,
        }
    promoters = (responses["NPS group"] == "Promoter").mean() * 100
    detractors = (responses["NPS group"] == "Detractor").mean() * 100
    return {
        "responses": float(responses["Response ID"].nunique()),
        "nps": promoters - detractors,
        "avg_nps": responses["NPS score"].mean(),
        "sat": responses["Satisfaction numeric"].mean(),
        "value_add": (responses["Any value-add flag"] == "Yes").mean() * 100,
        "follow_up": (responses["Follow-up permission flag"] == "Yes").mean() * 100,
    }


def top_theme_table(responses: pd.DataFrame) -> pd.DataFrame:
    themed_rows = []
    for _, row in responses[["Response ID", "Verbatim comment", "NPS group"]].dropna().iterrows():
        text = str(row["Verbatim comment"]).lower()
        matched = False
        for theme, keywords in THEME_MAP.items():
            if any(keyword in text for keyword in keywords):
                themed_rows.append(
                    {"Theme": theme, "Response ID": row["Response ID"], "NPS group": row["NPS group"]}
                )
                matched = True
        if not matched:
            themed_rows.append(
                {"Theme": "Other", "Response ID": row["Response ID"], "NPS group": row["NPS group"]}
            )

    if not themed_rows:
        return pd.DataFrame(columns=["Theme", "Mentions", "Detractor share"])

    themed = pd.DataFrame(themed_rows)
    summary = (
        themed.groupby("Theme")
        .agg(
            Mentions=("Response ID", "nunique"),
            Detractors=("NPS group", lambda s: (s == "Detractor").sum()),
        )
        .reset_index()
    )
    summary["Detractor share"] = summary["Detractors"] / summary["Mentions"] * 100
    return summary.sort_values(["Mentions", "Detractor share"], ascending=[False, False])


def benchmark_frame(responses: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = (
        responses.groupby(group_col, dropna=False)
        .apply(lambda df: pd.Series(summarize_scores(df)))
        .reset_index()
    )
    return grouped.sort_values("nps", ascending=False)


def make_trend_frame(responses: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    trend = (
        responses.groupby(group_cols + ["Month"], dropna=False)
        .apply(lambda df: pd.Series(summarize_scores(df)))
        .reset_index()
        .sort_values("Month")
    )
    return trend


def wait_time_impact_table(responses: pd.DataFrame) -> pd.DataFrame:
    service = responses[responses["Interaction type"] == "Service"].copy()
    if service.empty:
        return pd.DataFrame()

    wait_issue = service[service["Wait time issue flag"]]
    wait_ok = service[~service["Wait time issue flag"]]
    impact = (
        service.groupby(["Region", "Branch"])
        .agg(
            responses=("Response ID", "nunique"),
            wait_time_issue_rate=("Wait time issue flag", lambda s: s.mean() * 100),
            avg_nps=("NPS score", "mean"),
            sat=("Satisfaction numeric", "mean"),
        )
        .reset_index()
    )

    issue_nps = wait_issue.groupby(["Region", "Branch"])["NPS score"].mean().rename("issue_nps")
    ok_nps = wait_ok.groupby(["Region", "Branch"])["NPS score"].mean().rename("ok_nps")
    impact = impact.merge(issue_nps, on=["Region", "Branch"], how="left").merge(
        ok_nps, on=["Region", "Branch"], how="left"
    )
    impact["nps_gap_if_wait_issue"] = impact["ok_nps"] - impact["issue_nps"]
    return impact.sort_values(["nps_gap_if_wait_issue", "wait_time_issue_rate"], ascending=False)


def driver_frame(behaviour: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        behaviour.groupby(["Interaction type", "Behaviour statement"], dropna=False)
        .agg(
            avg_behaviour=("Behaviour score", "mean"),
            avg_nps=("NPS score", "mean"),
            detractor_rate=("NPS group", lambda s: (s == "Detractor").mean() * 100),
            responses=("Response ID", "nunique"),
        )
        .reset_index()
    )

    correlations = []
    for (interaction_type, statement), group in behaviour.groupby(
        ["Interaction type", "Behaviour statement"], dropna=False
    ):
        corr = group["Behaviour score"].corr(group["NPS score"])
        correlations.append(
            {
                "Interaction type": interaction_type,
                "Behaviour statement": statement,
                "nps_correlation": corr if not pd.isna(corr) else 0.0,
            }
        )
    corr_df = pd.DataFrame(correlations)
    merged = grouped.merge(corr_df, on=["Interaction type", "Behaviour statement"], how="left")
    merged["Behaviour short"] = merged["Behaviour statement"].map(lambda s: BEHAVIOUR_LABELS.get(s, s))
    return merged.sort_values("nps_correlation", ascending=False)


def add_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top right, rgba(248, 180, 98, 0.22), transparent 28%),
                    radial-gradient(circle at left 20%, rgba(8, 145, 178, 0.18), transparent 24%),
                    linear-gradient(180deg, #f7f5ef 0%, #fcfbf8 55%, #f1efe8 100%);
                color: #18212f;
            }
            .block-container {
                padding-top: 1.4rem;
                padding-bottom: 2.2rem;
            }
            .hero {
                background: linear-gradient(135deg, #133c55 0%, #0f6e8c 48%, #d17b0f 100%);
                border: 1px solid rgba(17, 24, 39, 0.08);
                border-radius: 24px;
                padding: 1.4rem 1.6rem;
                color: #fdfbf7;
                box-shadow: 0 20px 40px rgba(19, 60, 85, 0.12);
                margin-bottom: 1rem;
            }
            .hero h1 {
                margin: 0 0 0.4rem 0;
                font-size: 2.1rem;
            }
            .hero p {
                margin: 0.15rem 0;
                font-size: 1rem;
                opacity: 0.95;
            }
            .journey {
                display: inline-block;
                margin-top: 0.7rem;
                padding: 0.35rem 0.65rem;
                border-radius: 999px;
                background: rgba(255,255,255,0.15);
                font-size: 0.85rem;
            }
            .metric-card {
                background: rgba(255,255,255,0.78);
                border: 1px solid rgba(19, 60, 85, 0.08);
                border-radius: 18px;
                padding: 0.9rem 1rem;
                box-shadow: 0 12px 24px rgba(24, 33, 47, 0.06);
                min-height: 108px;
            }
            .metric-label {
                color: #355070;
                font-size: 0.86rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            .metric-value {
                color: #12263a;
                font-size: 1.8rem;
                font-weight: 700;
                margin-top: 0.35rem;
            }
            .metric-delta {
                color: #516173;
                font-size: 0.9rem;
                margin-top: 0.15rem;
            }
            .section-card {
                background: rgba(255,255,255,0.72);
                border: 1px solid rgba(19, 60, 85, 0.08);
                border-radius: 18px;
                padding: 1rem 1.1rem;
                margin-bottom: 1rem;
            }
            .pill {
                display: inline-block;
                padding: 0.25rem 0.55rem;
                border-radius: 999px;
                background: #eef4f7;
                color: #163042;
                border: 1px solid rgba(19, 60, 85, 0.08);
                font-size: 0.82rem;
                margin-right: 0.35rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_filters(responses: pd.DataFrame) -> tuple[str, dict[str, list[str]]]:
    st.sidebar.title("Dashboard Controls")
    role_view = st.sidebar.radio(
        "Stakeholder journey",
        list(ROLE_DESCRIPTIONS.keys()),
        help="The dashboard content shifts to the decisions and evidence needed for each audience.",
    )

    regions = ["All"] + sorted(responses["Region"].dropna().unique().tolist())
    selected_regions = st.sidebar.multiselect("Region", regions, default=["All"])

    branch_pool = responses.copy()
    if "All" not in selected_regions:
        branch_pool = branch_pool[branch_pool["Region"].isin(selected_regions)]
    branches = ["All"] + sorted(branch_pool["Branch"].dropna().unique().tolist())
    selected_branches = st.sidebar.multiselect("Branch", branches, default=["All"])

    staff_pool = branch_pool.copy()
    if "All" not in selected_branches:
        staff_pool = staff_pool[staff_pool["Branch"].isin(selected_branches)]
    staff_members = ["All"] + sorted(staff_pool["Staff member"].dropna().unique().tolist())
    selected_staff = st.sidebar.multiselect("Staff member", staff_members, default=["All"])

    roles = ["All"] + sorted(responses["Staff role"].dropna().unique().tolist())
    selected_roles = st.sidebar.multiselect("Staff role", roles, default=["All"])

    interaction_types = ["All"] + sorted(responses["Interaction type"].dropna().unique().tolist())
    selected_interactions = st.sidebar.multiselect(
        "Channel family",
        interaction_types,
        default=["All"],
        help="Service and Advice are the primary cross-dashboard comparison axis.",
    )

    appointment_types = ["All", "Service / Not applicable"] + sorted(
        responses["Appointment type"].dropna().unique().tolist()
    )
    selected_appointments = st.sidebar.multiselect(
        "Appointment type (advice only)",
        appointment_types,
        default=["All"],
    )

    min_responses = st.sidebar.slider("Minimum responses for tables", 1, 15, 3)
    st.sidebar.caption("Synthetic data note: all figures are fictional mock-up content, not actual member results.")

    filters = {
        "regions": selected_regions,
        "branches": selected_branches,
        "staff_members": selected_staff,
        "staff_roles": selected_roles,
        "interaction_types": selected_interactions,
        "appointment_types": selected_appointments,
        "min_responses": [min_responses],
    }
    return role_view, filters


def render_header(role_view: str, responses: pd.DataFrame, metadata: pd.DataFrame) -> None:
    dataset_label = metadata.loc[metadata["Item"] == "Dataset label", "Value"].iloc[0]
    date_min = responses["Survey date"].min().strftime("%b %d, %Y")
    date_max = responses["Survey date"].max().strftime("%b %d, %Y")
    st.markdown(
        f"""
        <div class="hero">
            <h1>Service & Advice Dashboard</h1>
            <p>{ROLE_DESCRIPTIONS[role_view]}</p>
            <p>{dataset_label} | Survey period: {date_min} to {date_max}</p>
            <span class="journey">signal → context → diagnosis → evidence → action</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview(responses: pd.DataFrame, benchmark_responses: pd.DataFrame) -> None:
    filtered_summary = summarize_scores(responses)
    org_summary = summarize_scores(benchmark_responses)

    st.subheader("Executive Snapshot")
    cols = st.columns(6)
    metric_card(cols[0], "Responses", f"{int(filtered_summary['responses'])}", f"Org {int(org_summary['responses'])}")
    metric_card(cols[1], "NPS", number(filtered_summary["nps"]), f"Org {number(org_summary['nps'])}")
    metric_card(cols[2], "Avg NPS score", number(filtered_summary["avg_nps"]), f"Org {number(org_summary['avg_nps'])}")
    metric_card(cols[3], "Satisfaction", number(filtered_summary["sat"]), f"Org {number(org_summary['sat'])}")
    metric_card(cols[4], "Any value-add", pct(filtered_summary["value_add"]), f"Org {pct(org_summary['value_add'])}")
    metric_card(cols[5], "Follow-up requested", pct(filtered_summary["follow_up"]), f"Org {pct(org_summary['follow_up'])}")


def render_trend_and_mix(responses: pd.DataFrame) -> None:
    trend = make_trend_frame(responses, ["Interaction type"])
    mix = (
        responses.groupby(["Interaction type", "Region"])
        .agg(responses=("Response ID", "nunique"))
        .reset_index()
    )

    left, right = st.columns((1.2, 0.8))
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Trend over time")
        if trend.empty:
            st.info("No trend data available for the current filter.")
        else:
            fig = px.line(
                trend,
                x="Month",
                y="nps",
                color="Interaction type",
                markers=True,
                labels={"nps": "NPS"},
            )
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Response mix")
        if mix.empty:
            st.info("No response mix available.")
        else:
            fig = px.bar(
                mix,
                x="Region",
                y="responses",
                color="Interaction type",
                barmode="stack",
            )
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_frontline_coaching(
    responses: pd.DataFrame,
    behaviour: pd.DataFrame,
    value_add: pd.DataFrame,
    min_responses: int,
) -> None:
    st.subheader("Frontline Coaching")
    st.caption("Decision support for: who needs coaching, what to coach on, and whether they are improving.")

    staff_table = (
        responses.groupby(["Staff member", "Staff role", "Region", "Branch", "Interaction type"], dropna=False)
        .apply(lambda df: pd.Series(summarize_scores(df)))
        .reset_index()
    )
    staff_table["low_volume"] = np.where(staff_table["responses"] < LOW_VOLUME_THRESHOLD, "Yes", "No")
    staff_table = staff_table[staff_table["responses"] >= min_responses].sort_values("nps")

    urgent = responses[responses["Negative experience flag"]].copy()
    urgent = urgent.sort_values(["Survey date", "NPS score"], ascending=[False, True])

    selected_staff = None
    if not staff_table.empty:
        selected_staff = st.selectbox(
            "Staff profile",
            staff_table["Staff member"].tolist(),
            index=0,
            help="Use this to pivot the diagnosis, verbatim evidence, and coaching suggestions to one person.",
        )

    col1, col2 = st.columns((1.15, 0.85))
    with col1:
        st.markdown("#### Staff scorecard")
        if staff_table.empty:
            st.info("No staff meet the current filter and minimum-response threshold.")
        else:
            show = staff_table.rename(
                columns={
                    "responses": "Responses",
                    "nps": "NPS",
                    "avg_nps": "Avg NPS score",
                    "sat": "Satisfaction",
                    "value_add": "Any value-add %",
                    "follow_up": "Follow-up %",
                    "low_volume": "Low volume",
                }
            )
            st.dataframe(
                show[
                    [
                        "Staff member",
                        "Staff role",
                        "Region",
                        "Branch",
                        "Interaction type",
                        "Responses",
                        "NPS",
                        "Satisfaction",
                        "Any value-add %",
                        "Follow-up %",
                        "Low volume",
                    ]
                ].round(1),
                use_container_width=True,
                hide_index=True,
            )

    with col2:
        st.markdown("#### Recent low-score alerts")
        if urgent.empty:
            st.success("No low-score or follow-up flagged responses within the current filter.")
        else:
            st.dataframe(
                urgent[
                    [
                        "Survey date",
                        "Staff member",
                        "Interaction type",
                        "Branch",
                        "NPS score",
                        "Satisfaction rating",
                        "Follow-up permission flag",
                        "Verbatim comment",
                    ]
                ].head(8),
                use_container_width=True,
                hide_index=True,
            )

    if selected_staff is None:
        return

    profile = responses[responses["Staff member"] == selected_staff].copy()
    profile_behaviour = behaviour[behaviour["Staff member"] == selected_staff].copy()
    profile_value = value_add[value_add["Response ID"].isin(profile["Response ID"])].copy()

    left, right = st.columns((0.9, 1.1))
    with left:
        st.markdown("#### Behaviour diagnosis")
        if profile_behaviour.empty:
            st.info("No behaviour detail available for this staff member.")
        else:
            behaviour_chart = (
                profile_behaviour.groupby("Behaviour statement")
                .agg(avg_score=("Behaviour score", "mean"), responses=("Response ID", "nunique"))
                .reset_index()
                .sort_values("avg_score")
            )
            behaviour_chart["Behaviour"] = behaviour_chart["Behaviour statement"].map(
                lambda s: BEHAVIOUR_LABELS.get(s, s)
            )
            fig = px.bar(
                behaviour_chart,
                x="avg_score",
                y="Behaviour",
                orientation="h",
                text="responses",
                color="avg_score",
                color_continuous_scale="Tealgrn",
                range_x=[1, 5],
            )
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Improvement trend")
        if profile.empty:
            st.info("No trend data available.")
        else:
            trend = make_trend_frame(profile, [])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trend["Month"], y=trend["nps"], mode="lines+markers", name="NPS"))
            fig.add_trace(
                go.Scatter(
                    x=trend["Month"],
                    y=trend["sat"],
                    mode="lines+markers",
                    name="Satisfaction",
                    yaxis="y2",
                )
            )
            fig.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(title="NPS"),
                yaxis2=dict(title="Satisfaction", overlaying="y", side="right", range=[1, 5]),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns((0.8, 1.2))
    with left:
        st.markdown("#### Value-add outcomes")
        if profile_value.empty:
            st.info("No value-add selections available.")
        else:
            outcomes = (
                profile_value.groupby("Value-add outcome selected")
                .agg(selections=("Response ID", "count"))
                .reset_index()
                .sort_values("selections", ascending=False)
            )
            fig = px.bar(outcomes, x="selections", y="Value-add outcome selected", orientation="h")
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Verbatim evidence")
        st.dataframe(
            profile[
                [
                    "Survey date",
                    "NPS group",
                    "Satisfaction rating",
                    "Any value-add flag",
                    "Verbatim comment",
                ]
            ]
            .sort_values("Survey date", ascending=False)
            .head(8),
            use_container_width=True,
            hide_index=True,
        )

    interaction_mix = (
        responses.groupby(["Branch", "Interaction type"])
        .apply(lambda df: pd.Series(summarize_scores(df)))
        .reset_index()
    )
    branch_name = profile["Branch"].iloc[0]
    comparison = interaction_mix[interaction_mix["Branch"] == branch_name]
    st.markdown("#### Service vs advice context")
    if comparison["Interaction type"].nunique() < 2:
        st.info(
            "The selected staff member only appears in one interaction type. This dataset supports branch/team-level service vs advice comparison rather than true per-person cross-channel comparison."
        )
    else:
        fig = px.bar(
            comparison,
            x="Interaction type",
            y="nps",
            color="Interaction type",
            text="responses",
            labels={"nps": "NPS"},
        )
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def render_branch_performance(
    responses: pd.DataFrame,
    behaviour: pd.DataFrame,
    value_add: pd.DataFrame,
    min_responses: int,
) -> None:
    st.subheader("Branch Performance")
    st.caption("Decision support for: where the branch is underperforming and what to prioritize next.")

    branch_summary = benchmark_frame(responses, "Branch")
    branch_summary = branch_summary[branch_summary["responses"] >= min_responses]
    if branch_summary.empty:
        st.info("No branch-level summary available.")
        return

    selected_branch = st.selectbox("Branch benchmark", branch_summary["Branch"].tolist())
    branch_data = responses[responses["Branch"] == selected_branch].copy()
    region_name = branch_data["Region"].iloc[0]
    region_data = responses[responses["Region"] == region_name].copy()
    org_data = responses.copy()

    comparisons = pd.DataFrame(
        [
            {"Level": selected_branch, **summarize_scores(branch_data)},
            {"Level": region_name, **summarize_scores(region_data)},
            {"Level": ORG_LABEL, **summarize_scores(org_data)},
        ]
    )
    left, right = st.columns((0.85, 1.15))
    with left:
        st.markdown("#### Benchmark ladder")
        fig = px.bar(
            comparisons,
            x="nps",
            y="Level",
            orientation="h",
            color="Level",
            text="responses",
            labels={"nps": "NPS"},
        )
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Branch trend")
        trend = make_trend_frame(branch_data, ["Interaction type"])
        fig = px.line(
            trend,
            x="Month",
            y="nps",
            color="Interaction type",
            markers=True,
            labels={"nps": "NPS"},
        )
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Channel comparison")
        channel = (
            branch_data.groupby("Interaction type")
            .apply(lambda df: pd.Series(summarize_scores(df)))
            .reset_index()
        )
        fig = px.bar(
            channel,
            x="Interaction type",
            y=["nps", "value_add"],
            barmode="group",
            labels={"value": "Score", "variable": "Metric"},
        )
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Staff reliability context")
        staff_counts = (
            branch_data.groupby(["Staff member", "Interaction type"])
            .agg(responses=("Response ID", "nunique"), nps=("NPS score", "mean"))
            .reset_index()
        )
        fig = px.scatter(
            staff_counts,
            x="responses",
            y="nps",
            color="Interaction type",
            text="Staff member",
            size="responses",
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Value-add outcomes")
        branch_values = value_add[value_add["Response ID"].isin(branch_data["Response ID"])].copy()
        if branch_values.empty:
            st.info("No value-add data available.")
        else:
            outcomes = (
                branch_values.groupby("Value-add outcome selected")
                .agg(selections=("Response ID", "count"))
                .reset_index()
                .sort_values("selections", ascending=False)
            )
            fig = px.bar(outcomes, x="selections", y="Value-add outcome selected", orientation="h")
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Branch behaviour profile")
        branch_behaviour = behaviour[behaviour["Branch"] == selected_branch].copy()
        if branch_behaviour.empty:
            st.info("No behaviour profile available.")
        else:
            heat = (
                branch_behaviour.groupby(["Interaction type", "Behaviour statement"])
                .agg(avg_score=("Behaviour score", "mean"))
                .reset_index()
            )
            heat["Behaviour"] = heat["Behaviour statement"].map(lambda s: BEHAVIOUR_LABELS.get(s, s))
            pivot = heat.pivot(index="Behaviour", columns="Interaction type", values="avg_score")
            fig = px.imshow(
                pivot.fillna(np.nan),
                aspect="auto",
                color_continuous_scale="Brwnyl",
                zmin=1,
                zmax=5,
            )
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), coloraxis_colorbar_title="Score")
            st.plotly_chart(fig, use_container_width=True)


def render_regional_leadership(
    responses: pd.DataFrame,
    behaviour: pd.DataFrame,
    value_add: pd.DataFrame,
    min_responses: int,
) -> None:
    st.subheader("Regional Leadership")
    st.caption("Decision support for: which branches need help, where to allocate support, and what patterns are systemic.")

    region_summary = benchmark_frame(responses, "Region")
    if region_summary.empty:
        st.info("No regional summary available.")
        return

    selected_region = st.selectbox("Region focus", region_summary["Region"].tolist())
    region_data = responses[responses["Region"] == selected_region].copy()

    left, right = st.columns((1.1, 0.9))
    with left:
        st.markdown("#### Branch comparison")
        branches = benchmark_frame(region_data, "Branch")
        branches = branches[branches["responses"] >= min_responses]
        fig = px.scatter(
            branches,
            x="responses",
            y="nps",
            size="value_add",
            color="follow_up",
            text="Branch",
            color_continuous_scale="RdYlGn_r",
            labels={"nps": "NPS", "follow_up": "Follow-up %", "value_add": "Value-add %"},
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Regional trend")
        trend = make_trend_frame(region_data, ["Interaction type"])
        fig = px.line(
            trend,
            x="Month",
            y="sat",
            color="Interaction type",
            markers=True,
            labels={"sat": "Satisfaction"},
        )
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_range=[1, 5])
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Wait time risk branches")
        wait_impact = wait_time_impact_table(region_data)
        if wait_impact.empty:
            st.info("Wait time analysis applies only to service interactions.")
        else:
            st.dataframe(
                wait_impact[
                    ["Branch", "responses", "wait_time_issue_rate", "issue_nps", "ok_nps", "nps_gap_if_wait_issue"]
                ]
                .rename(
                    columns={
                        "responses": "Responses",
                        "wait_time_issue_rate": "Wait issue %",
                        "issue_nps": "NPS when wait issue",
                        "ok_nps": "NPS when wait acceptable",
                        "nps_gap_if_wait_issue": "NPS gap",
                    }
                )
                .round(1),
                use_container_width=True,
                hide_index=True,
            )

    with right:
        st.markdown("#### Segmentation by role and appointment")
        segment = (
            region_data.groupby(["Staff role", "Appointment type"], dropna=False)
            .apply(lambda df: pd.Series(summarize_scores(df)))
            .reset_index()
        )
        segment["Appointment type"] = segment["Appointment type"].fillna("Service / Not applicable")
        fig = px.scatter(
            segment,
            x="sat",
            y="nps",
            size="responses",
            color="Staff role",
            hover_name="Appointment type",
            labels={"sat": "Satisfaction", "nps": "NPS"},
        )
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns((0.95, 1.05))
    with left:
        st.markdown("#### Value-add by branch")
        region_values = value_add[value_add["Response ID"].isin(region_data["Response ID"])].copy()
        if region_values.empty:
            st.info("No value-add data available.")
        else:
            table = (
                region_values[region_values["Any value-add flag"] == "Yes"]
                .groupby("Branch")
                .agg(selections=("Response ID", "nunique"))
                .reset_index()
                .sort_values("selections", ascending=False)
            )
            fig = px.bar(table, x="Branch", y="selections", color="Branch")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Verbatim themes")
        themes = top_theme_table(region_data)
        st.dataframe(themes.round(1), use_container_width=True, hide_index=True)


def render_mx_integration(
    responses: pd.DataFrame,
    behaviour: pd.DataFrame,
    value_add: pd.DataFrame,
) -> None:
    st.subheader("MX Integration & Member Discovery")
    st.caption("Decision support for: what drives NPS and satisfaction, where to invest, and which patterns define stronger performance.")

    drivers = driver_frame(behaviour)

    left, right = st.columns((1.1, 0.9))
    with left:
        st.markdown("#### Behaviour drivers of NPS")
        if drivers.empty:
            st.info("No driver analysis available.")
        else:
            fig = px.scatter(
                drivers,
                x="avg_behaviour",
                y="nps_correlation",
                color="Interaction type",
                size="responses",
                hover_name="Behaviour short",
                labels={"avg_behaviour": "Average behaviour score", "nps_correlation": "Correlation with NPS"},
            )
            fig.add_vline(x=4, line_dash="dot", line_color="#355070")
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Top drivers table")
        st.dataframe(
            drivers[
                [
                    "Interaction type",
                    "Behaviour short",
                    "avg_behaviour",
                    "nps_correlation",
                    "detractor_rate",
                    "responses",
                ]
            ]
            .rename(
                columns={
                    "Behaviour short": "Behaviour",
                    "avg_behaviour": "Avg score",
                    "nps_correlation": "NPS correlation",
                    "detractor_rate": "Detractor %",
                    "responses": "Responses",
                }
            )
            .round(2)
            .head(12),
            use_container_width=True,
            hide_index=True,
        )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Branch performance map")
        branches = (
            responses.groupby(["Region", "Branch"], dropna=False)
            .apply(lambda df: pd.Series(summarize_scores(df)))
            .reset_index()
        )
        fig = px.scatter(
            branches,
            x="value_add",
            y="nps",
            color="Region",
            size="responses",
            text="Branch",
            labels={"value_add": "Value-add %", "nps": "NPS"},
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Appointment and role segmentation")
        segment = (
            responses.groupby(["Interaction type", "Staff role", "Appointment type"], dropna=False)
            .apply(lambda df: pd.Series(summarize_scores(df)))
            .reset_index()
        )
        segment["Appointment type"] = segment["Appointment type"].fillna("Service / Not applicable")
        fig = px.treemap(
            segment,
            path=["Interaction type", "Staff role", "Appointment type"],
            values="responses",
            color="nps",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
        )
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Signal-to-action summary")
    themes = top_theme_table(responses)
    strongest = drivers.sort_values("nps_correlation", ascending=False).head(3)["Behaviour short"].tolist()
    weakest = drivers.sort_values("avg_behaviour", ascending=True).head(3)["Behaviour short"].tolist()
    top_themes = themes.head(3)["Theme"].tolist()

    st.markdown(
        f"""
        <div class="section-card">
            <span class="pill">Top drivers: {", ".join(strongest) if strongest else "n/a"}</span>
            <span class="pill">Lowest scoring behaviours: {", ".join(weakest) if weakest else "n/a"}</span>
            <span class="pill">Common verbatim themes: {", ".join(top_themes) if top_themes else "n/a"}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    add_styles()
    data = load_data()
    all_responses = data["responses"]
    role_view, filters = sidebar_filters(all_responses)

    responses = apply_filters(
        all_responses,
        regions=filters["regions"],
        branches=filters["branches"],
        staff_members=filters["staff_members"],
        staff_roles=filters["staff_roles"],
        interaction_types=filters["interaction_types"],
        appointment_types=filters["appointment_types"],
    )
    behaviour = apply_filters(
        data["behaviour"],
        regions=filters["regions"],
        branches=filters["branches"],
        staff_members=filters["staff_members"],
        staff_roles=filters["staff_roles"],
        interaction_types=filters["interaction_types"],
        appointment_types=filters["appointment_types"],
    )
    value_add = data["value_add"][data["value_add"]["Response ID"].isin(responses["Response ID"])].copy()

    render_header(role_view, responses if not responses.empty else all_responses, data["metadata"])
    if responses.empty:
        st.warning("The current filter combination returned no responses. Adjust the sidebar filters.")
        return

    render_overview(responses, all_responses)
    render_trend_and_mix(responses)

    if role_view == "Frontline Coaching":
        render_frontline_coaching(responses, behaviour, value_add, filters["min_responses"][0])
    elif role_view == "Branch Performance":
        render_branch_performance(responses, behaviour, value_add, filters["min_responses"][0])
    elif role_view == "Regional Leadership":
        render_regional_leadership(responses, behaviour, value_add, filters["min_responses"][0])
    else:
        render_mx_integration(responses, behaviour, value_add)

    st.caption(
        "Synthetic mock-up disclaimer: this dashboard uses illustrative fictional responses derived from the uploaded survey drafts and user-story requirements."
    )


if __name__ == "__main__":
    main()
