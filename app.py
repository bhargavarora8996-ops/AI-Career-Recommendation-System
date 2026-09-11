"""
app.py
Streamlit Web Application for AI Student Career Intelligence & Recommendation System.
Features Two-Level Career Intelligence (Level 1 ML Domain Classification + Level 2 Top-K Compatibility Ranking).
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import importlib
import src.preprocessing as prep
try:
    importlib.reload(prep)
except Exception:
    pass

def fallback_clean_delimited_text(text):
    if pd.isna(text) or not str(text).strip():
        return ''
    tokens = [t.strip().lower() for t in str(text).split(';') if t.strip()]
    return ' '.join(tokens)

def fallback_parse_delimited_list(text):
    if pd.isna(text) or not str(text).strip():
        return []
    return [t.strip() for t in str(text).split(';') if t.strip()]

load_dataset = getattr(prep, 'load_dataset', None)
clean_delimited_text = getattr(prep, 'clean_delimited_text', fallback_clean_delimited_text)
map_career_to_domain = getattr(prep, 'map_career_to_domain', None)
parse_delimited_list = getattr(prep, 'parse_delimited_list', fallback_parse_delimited_list)
import src.eda as eda
import src.prediction as pred
import src.explainability as exp_mod
import src.recommendations as rec

for m in [prep, eda, pred, exp_mod, rec]:
    try:
        importlib.reload(m)
    except Exception:
        pass

get_dataset_overview = eda.get_dataset_overview
get_top_skills = eda.get_top_skills
get_top_interests = eda.get_top_interests
create_career_dist_chart = eda.create_career_dist_chart
create_education_dist_chart = eda.create_education_dist_chart
create_skills_chart = eda.create_skills_chart
create_interests_chart = eda.create_interests_chart
create_age_dist_chart = eda.create_age_dist_chart

predict_career = pred.predict_career
explain_prediction = exp_mod.explain_prediction

rank_topk_recommendations = rec.rank_topk_recommendations
generate_top3_comparison_matrix = rec.generate_top3_comparison_matrix
analyze_skill_gap = rec.analyze_skill_gap
generate_learning_roadmap = rec.generate_learning_roadmap
get_project_recommendations = rec.get_project_recommendations


# Page Configuration
st.set_page_config(
    page_title="AI Student Career Intelligence System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern UI
st.markdown("""
<style>
    .main { background-color: #F8FAFC; }
    
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 2px;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 500;
    }

    .recommendation-box {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        color: white;
        border-radius: 12px;
        padding: 22px;
        box-shadow: 0 10px 15px -3px rgba(30, 58, 138, 0.3);
        margin-bottom: 20px;
    }
    
    .career-rank-card {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 15px;
        border-left: 6px solid #3B82F6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
    }

    .warning-banner {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 20px;
        font-size: 0.92rem;
        color: #92400E;
    }

    .skill-tag {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        border-radius: 6px;
        padding: 3px 9px;
        font-size: 0.83rem;
        font-weight: 600;
        margin: 2px;
    }

    .missing-tag {
        display: inline-block;
        background-color: #FEF2F2;
        color: #DC2626;
        border: 1px solid #FECACA;
        border-radius: 6px;
        padding: 3px 9px;
        font-size: 0.83rem;
        font-weight: 600;
        margin: 2px;
    }

    .step-card {
        background-color: #FFFFFF;
        border-left: 4px solid #3B82F6;
        padding: 14px 18px;
        margin-bottom: 12px;
        border-radius: 4px 8px 8px 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_app_dataset():
    return load_dataset()


@st.cache_data
def load_comp_df(target_type='domain'):
    path = os.path.join('models', f'model_comparison_{target_type}.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_topk_metrics(target_type='32'):
    path = os.path.join('models', f'topk_metrics_{target_type}.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def main():
    try:
        df, dataset_path = load_app_dataset()
        df['Career_Domain'] = df['Recommended_Career'].apply(map_career_to_domain)

        all_skills_set = set()
        all_interests_set = set()
        for s in df['Skills'].dropna():
            all_skills_set.update(parse_delimited_list(s))
        for i in df['Interests'].dropna():
            all_interests_set.update(parse_delimited_list(i))

        all_skills = sorted(list(all_skills_set))
        all_interests = sorted(list(all_interests_set))
        all_education = sorted(list(df['Education'].unique()))
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

    # Sidebar Navigation
    st.sidebar.image("https://img.icons8.com/isometric/100/graduation-cap.png", width=65)
    st.sidebar.title("Career Intelligence")
    st.sidebar.caption("Two-Level Recommendation Platform")

    menu_option = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Home & Overview",
            "🎯 Career Predictor & Recommendations",
            "⚖️ Top Career Comparison",
            "🎯 Skill Gap Analysis",
            "🗺️ Personalized Roadmap & Projects",
            "📊 Dataset Insights & EDA",
            "📈 Model Performance & Top-K Metrics",
            "📖 About & Reliability Warnings"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.info(f"**Dataset:** `{os.path.basename(dataset_path)}`\n\n**Candidates:** {len(df)} rows")

    # Session State
    if 'domain_pred' not in st.session_state:
        st.session_state.domain_pred = None
    if 'topk_recs' not in st.session_state:
        st.session_state.topk_recs = None

    comp_dom = load_comp_df('domain')
    comp_32 = load_comp_df('32')
    topk_32 = load_topk_metrics('32')

    best_dom_model = comp_dom.iloc[0]['Model'] if comp_dom is not None else "Logistic Regression (balanced)"
    best_dom_acc = comp_dom.iloc[0]['Test Accuracy'] if comp_dom is not None else 0.575
    best_dom_macro_f1 = comp_dom.iloc[0]['Macro F1'] if comp_dom is not None else 0.536

    # ==========================================
    # 🏠 HOME & OVERVIEW
    # ==========================================
    if menu_option == "🏠 Home & Overview":
        st.title("🎓 AI Student Career Intelligence System")
        st.subheader("Data-Driven Two-Level Career Recommendation Platform")

        # Reliability Banner
        st.markdown("""
        <div class="warning-banner">
            <strong>⚠️ Model Reliability Note:</strong> This system provides data-driven career recommendations based on the available dataset. It is intended as a decision-support tool and not as a definitive career predictor.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        Welcome to the **AI Student Career Intelligence System**. This system operates via a **Two-Level Architecture**:
        - **Level 1 (Supervised ML Classifier):** Predicts the primary **Career Domain** (e.g. *Data Science & AI*, *Software & Systems*) using Scikit-Learn pipelines.
        - **Level 2 (Career Recommendation Engine):** Computes transparent **Career Compatibility Scores (0–100)** ranking specific fine-grained careers based on skill similarity, interest match, education compatibility, and domain confidence.
        """)

        st.markdown("---")
        st.markdown("### 📌 Core System Metrics")

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(df)}</div>
                <div class="metric-label">Students in Dataset</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">7</div>
                <div class="metric-label">Career Domains</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">32</div>
                <div class="metric-label">Fine-Grained Careers</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{best_dom_acc*100:.1f}%</div>
                <div class="metric-label">Domain ML Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with c5:
            top3_acc_val = topk_32.get('Top-3 Accuracy', 0.55) * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{top3_acc_val:.1f}%</div>
                <div class="metric-label">Top-3 Career Accuracy</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🚀 System Capabilities & Workflow")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("""
            #### Level 1: ML Domain Classification
            - Trains 7 classifiers using leak-free `Pipeline` objects.
            - Evaluates models on 7 broader career domains.
            - Achieves **57.5% Accuracy** & **0.536 Macro F1**.
            """)
        with col_b:
            st.markdown("""
            #### Level 2: Top-K Ranking Engine
            - Ranks fine-grained careers using transparent 4-factor formula.
            - Evaluates **Top-1 (42.5%)**, **Top-3 (55.0%)**, and **Top-5 (60.0%)** accuracy.
            - Mitigates title boundary overlap.
            """)
        with col_c:
            st.markdown("""
            #### Skill Gap & Learning Roadmap
            - Derives career skill baselines directly from dataset.
            - Highlights present vs missing required skills.
            - Generates rule-based learning pathways & project ideas.
            """)

        st.markdown("---")
        st.success("👉 Select **🎯 Career Predictor & Recommendations** in the sidebar to get started!")

    # ==========================================
    # 🎯 CAREER PREDICTOR & RECOMMENDATIONS
    # ==========================================
    elif menu_option == "🎯 Career Predictor & Recommendations":
        st.title("🎯 Two-Level Career Recommendation Engine")
        st.markdown("Enter the student's profile details below to generate Level 1 Domain predictions and Level 2 Ranked Career Recommendations.")

        with st.form(key="predictor_form"):
            col_a, col_b = st.columns([1, 2])

            with col_a:
                age_val = st.slider("Student Age:", 18, 50, 24)
                edu_val = st.selectbox("Education Level:", options=all_education, index=0)

            with col_b:
                skills_val = st.multiselect(
                    "Current Skills (Select all that apply):",
                    options=all_skills,
                    default=["python", "machine learning", "sql", "data analysis"] if "python" in all_skills else all_skills[:3]
                )
                interests_val = st.multiselect(
                    "Career Interests (Select all that apply):",
                    options=all_interests,
                    default=["technology", "data science"] if "technology" in all_interests else all_interests[:2]
                )

            submit_btn = st.form_submit_button(label="🔮 Analyze Profile & Generate Recommendations", use_container_width=True)

        if submit_btn:
            if not skills_val and not interests_val:
                st.warning("Please select at least one skill or interest.")
            else:
                with st.spinner("Executing Level 1 ML Classifier & Level 2 Recommendation Engine..."):
                    try:
                        dom_pred = predict_career(age_val, edu_val, skills_val, interests_val, target_type='domain')
                        topk_recs = rank_topk_recommendations(age_val, edu_val, skills_val, interests_val, domain_pred_res=dom_pred, top_k=5)

                        st.session_state.domain_pred = dom_pred
                        st.session_state.topk_recs = topk_recs
                        st.session_state.student_inputs = {
                            'age': age_val,
                            'education': edu_val,
                            'skills': skills_val,
                            'interests': interests_val
                        }
                    except Exception as err:
                        st.error(f"Execution Error: {err}")

        # Display Results
        if st.session_state.domain_pred is not None and st.session_state.topk_recs is not None:
            dom_res = st.session_state.domain_pred
            topk_recs = st.session_state.topk_recs
            inputs = st.session_state.student_inputs

            st.markdown("---")

            # LEVEL 1 DISPLAY
            st.markdown("### LEVEL 1: CAREER DOMAIN PREDICTION (Supervised ML Model)")
            st.markdown(f"""
            <div class="recommendation-box">
                <div class="recommendation-title">Predicted Domain: 🎯 {dom_res['top_career']}</div>
                <div class="recommendation-sub">Level 1 ML Model Confidence: <b>{dom_res['top_confidence']:.2f}%</b> (Classifier: {best_dom_model})</div>
            </div>
            """, unsafe_allow_html=True)

            # LEVEL 2 DISPLAY
            st.markdown("### LEVEL 2: TOP CAREER RECOMMENDATIONS (Recommendation Engine)")
            st.caption("Ranked using transparent 4-factor scoring formula: 45% Skill Match + 30% Interest Match + 15% Education Match + 10% Domain ML Fit.")

            for item in topk_recs[:3]:
                limited_badge = f' <span style="color:#DC2626; font-size:0.8rem; font-weight:bold;">[{item["sample_warning"]}]</span>' if item['is_limited_samples'] else ''
                st.markdown(f"""
                <div class="career-rank-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h3 style="margin:0; color:#1E3A8A;">{item['badge']} {item['career_title']}{limited_badge}</h3>
                        <div style="font-size:1.6rem; font-weight:800; color:#2563EB;">Score: {item['compatibility_score']} / 100</div>
                    </div>
                    <p style="margin:5px 0 10px 0; color:#64748B; font-weight:500;">Domain: <b>{item['career_domain']}</b> | Dataset Candidates: <b>{item['sample_count']}</b></p>
                </div>
                """, unsafe_allow_html=True)

                col_w1, col_w2 = st.columns([1, 1])
                with col_w1:
                    st.markdown("**Matching Skills:**")
                    if item['matched_skills']:
                        st.markdown("".join([f'<span class="skill-tag">{s}</span>' for s in item['matched_skills']]), unsafe_allow_html=True)
                    else:
                        st.info("No direct skill match found.")

                    st.markdown("**Skills to Develop:**")
                    if item['missing_skills']:
                        st.markdown("".join([f'<span class="missing-tag">{s}</span>' for s in item['missing_skills']]), unsafe_allow_html=True)

                with col_w2:
                    st.markdown("**Why This Recommendation?**")
                    for bullet in item['why_bullets']:
                        st.markdown(bullet)

                st.markdown("<hr style='margin:15px 0;'>", unsafe_allow_html=True)

    # ==========================================
    # ⚖️ TOP CAREER COMPARISON
    # ==========================================
    elif menu_option == "⚖️ Top Career Comparison":
        st.title("⚖️ Top 3 Career Comparison Matrix")
        st.markdown("Side-by-side comparison of the top 3 recommended careers across skill, interest, education, and domain compatibility factors.")

        if st.session_state.topk_recs is not None:
            top3 = st.session_state.topk_recs[:3]
            comp_matrix = generate_top3_comparison_matrix(top3)

            st.markdown("### 📊 Side-by-Side Comparison Table")
            st.dataframe(comp_matrix.style.highlight_max(subset=['Overall Compatibility Score'], color='#D1FAE5'), use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📈 Factor Score Breakdown Visualization")

            chart_data = []
            for item in top3:
                chart_data.extend([
                    {'Career': item['career_title'], 'Factor': 'Skill Match %', 'Score': item['skill_match_pct']},
                    {'Career': item['career_title'], 'Factor': 'Interest Match %', 'Score': item['interest_match_pct']},
                    {'Career': item['career_title'], 'Factor': 'Education Match %', 'Score': item['education_match_pct']},
                    {'Career': item['career_title'], 'Factor': 'Domain Fit %', 'Score': item['domain_fit_pct']}
                ])
            chart_df = pd.DataFrame(chart_data)

            fig_comp = px.bar(
                chart_df,
                x='Factor',
                y='Score',
                color='Career',
                barmode='group',
                title='Match Factors Comparison Across Top 3 Careers',
                height=400
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info("👉 Run a prediction first in **🎯 Career Predictor & Recommendations** to view comparison data.")

    # ==========================================
    # 🎯 SKILL GAP ANALYSIS
    # ==========================================
    elif menu_option == "🎯 Skill Gap Analysis":
        st.title("🎯 Skill Gap Analysis Engine")
        st.markdown("Compares student's skills against dataset career profiles.")

        target_career_sel = st.selectbox(
            "Select Target Career to Analyze:",
            options=sorted(df['Recommended_Career'].unique())
        )

        if st.session_state.student_inputs is not None:
            user_skills = st.session_state.student_inputs['skills']
        else:
            user_skills = st.multiselect(
                "Candidate Skills:",
                options=all_skills,
                default=["python", "sql", "data analysis"] if "python" in all_skills else all_skills[:2]
            )

        gap = analyze_skill_gap(target_career_sel, user_skills)

        st.markdown("---")
        cm1, cm2 = st.columns([1, 2])

        with cm1:
            st.markdown("### Skill Match Score")
            score = gap['match_score']
            st.metric(label=f"Match Score for {target_career_sel}", value=f"{score}%")
            st.progress(score / 100.0)

            if gap['is_limited_samples']:
                st.warning("⚠️ **Limited training examples:** Dataset contains <= 4 candidates for this specific career.")

        with cm2:
            st.markdown("### Skill Breakdown")
            st.markdown("**Matched Skills (Already possessed):**")
            if gap['matched_skills']:
                st.markdown("".join([f'<span class="skill-tag">{s}</span>' for s in gap['matched_skills']]), unsafe_allow_html=True)
            else:
                st.info("No matching skills found in baseline profile.")

            st.markdown("<br>**Skills to Develop (Recommended):**", unsafe_allow_html=True)
            if gap['missing_skills']:
                st.markdown("".join([f'<span class="missing-tag">{s}</span>' for s in gap['missing_skills']]), unsafe_allow_html=True)
            else:
                st.success("Candidate possesses all primary dataset skills for this career!")

    # ==========================================
    # 🗺️ PERSONALIZED ROADMAP & PROJECTS
    # ==========================================
    elif menu_option == "🗺️ Personalized Roadmap & Projects":
        st.title("🗺️ Personalized Learning Roadmap & Project Suggestions")

        if st.session_state.topk_recs is not None:
            t_car = st.session_state.topk_recs[0]['career_title']
            s_sk = st.session_state.student_inputs['skills']
            gap = analyze_skill_gap(t_car, s_sk)
            m_sk = gap['missing_skills']
        else:
            t_car = "Data Scientist"
            s_sk = ["python", "sql"]
            m_sk = ["machine learning", "statistics", "data visualization"]

        st.markdown(f"### 📍 Rule-Based Personalized Learning Roadmap for: **{t_car}**")
        st.caption("Note: Rule-Based Personalized Learning Roadmap derived from dataset skill gap analysis.")

        steps = generate_learning_roadmap(t_car, s_sk, m_sk)
        for st_item in steps:
            st.markdown(f"""
            <div class="step-card">
                <h4 style="margin:0; color:#1E3A8A;">{st_item['step']}: {st_item['title']}</h4>
                <p style="margin:4px 0 0 0; color:#475569;">{st_item['description']}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"### 📁 Project Suggestions for {t_car}")
        st.caption("Note: Project Suggestions tailored to target career domain.")

        projs = get_project_recommendations(t_car)
        p1, p2, p3 = st.columns(3)
        for idx, pr in enumerate(projs):
            col_target = [p1, p2, p3][idx % 3]
            with col_target:
                st.markdown(f"""
                <div class="metric-card" style="text-align:left;">
                    <h4 style="color:#1E3A8A; margin-top:0;">{pr['title']}</h4>
                    <p style="color:#475569; font-size:0.88rem;">{pr['description']}</p>
                    <hr style="margin:8px 0;">
                    <span style="font-size:0.8rem; color:#2563EB; font-weight:600;">Tech Stack: {pr['technologies']}</span>
                </div>
                """, unsafe_allow_html=True)

    # ==========================================
    # 📊 DATASET INSIGHTS & EDA
    # ==========================================
    elif menu_option == "📊 Dataset Insights & EDA":
        st.title("📊 Dataset Insights & Visual Exploratory Analysis")
        st.markdown("Visual insights derived from the 200 candidate Kaggle dataset.")

        tb1, tb2, tb3, tb4 = st.tabs([
            "🎯 Category & Domain Split",
            "🎓 Education & Age",
            "💡 Top Skills",
            "❤️ Top Interests"
        ])

        with tb1:
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.plotly_chart(create_career_dist_chart(df), use_container_width=True)
            with col_c2:
                dom_counts = df['Career_Domain'].value_counts().reset_index()
                dom_counts.columns = ['Domain', 'Count']
                fig_dom = px.pie(
                    dom_counts,
                    names='Domain',
                    values='Count',
                    title='Distribution across 7 Career Domains',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_dom.update_traces(textinfo='percent+label')
                st.plotly_chart(fig_dom, use_container_width=True)

        with tb2:
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                st.plotly_chart(create_education_dist_chart(df), use_container_width=True)
            with col_e2:
                st.plotly_chart(create_age_dist_chart(df), use_container_width=True)

        with tb3:
            n_sk = st.slider("Top N Skills:", 5, 30, 15)
            st.plotly_chart(create_skills_chart(df, top_n=n_sk), use_container_width=True)

        with tb4:
            n_in = st.slider("Top N Interests:", 5, 30, 15)
            st.plotly_chart(create_interests_chart(df, top_n=n_in), use_container_width=True)

    # ==========================================
    # 📈 MODEL PERFORMANCE & TOP-K METRICS
    # ==========================================
    elif menu_option == "📈 Model Performance & Top-K Metrics":
        st.title("📈 Model Benchmarking & Top-K Accuracy Evaluation")

        tab_mdom, tab_m32 = st.tabs(["🌐 Level 1: 7 Career Domains", "📊 Level 2 / Fine-Grained: 32 Careers"])

        with tab_mdom:
            if comp_dom is not None:
                st.markdown("### Level 1 Model Comparison Table (7 Career Domains)")
                st.dataframe(comp_dom.style.highlight_max(subset=['Test Accuracy', 'Macro F1', 'Weighted F1'], color='#D1FAE5'), use_container_width=True)

                col_bd1, col_bd2 = st.columns(2)
                with col_bd1:
                    fig_cd1 = px.bar(comp_dom, x='Model', y='Macro F1', color='Macro F1', title='Macro F1 Score (7 Domains)', text='Macro F1', color_continuous_scale='Viridis')
                    fig_cd1.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                    st.plotly_chart(fig_cd1, use_container_width=True)

                with col_bd2:
                    fig_cd2 = px.bar(comp_dom, x='Model', y='Test Accuracy', color='Test Accuracy', title='Test Accuracy (7 Domains)', text='Test Accuracy', color_continuous_scale='Cividis')
                    fig_cd2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                    st.plotly_chart(fig_cd2, use_container_width=True)

                # 7 Domain Confusion Matrix
                cm_dom_path = os.path.join('models', 'confusion_matrix_domain.npy')
                if os.path.exists(cm_dom_path):
                    cm_d = np.load(cm_dom_path)
                    st.markdown("### 🧩 Level 1 Domain Confusion Matrix Heatmap")
                    fig_cm_d = px.imshow(
                        cm_d,
                        labels=dict(x="Predicted Domain", y="Actual Domain"),
                        title="Confusion Matrix (7 Career Domains)",
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig_cm_d, use_container_width=True)

        with tab_m32:
            if comp_32 is not None:
                st.markdown("### Fine-Grained Model Comparison Table (32 Careers)")
                st.dataframe(comp_32.style.highlight_max(subset=['Test Accuracy', 'Macro F1', 'Top-3 Acc', 'Top-5 Acc'], color='#D1FAE5'), use_container_width=True)

                st.markdown("### 🏆 Top-K Ranking Accuracy Panel (32 Careers)")
                k1, k3, k5 = st.columns(3)
                with k1:
                    st.metric("Top-1 Accuracy", f"{topk_32.get('Top-1 Accuracy', 0.425)*100:.1f}%", help="Correct career is #1 prediction")
                with k3:
                    st.metric("Top-3 Accuracy", f"{topk_32.get('Top-3 Accuracy', 0.55)*100:.1f}%", help="Correct career appears in top 3 predictions")
                with k5:
                    st.metric("Top-5 Accuracy", f"{topk_32.get('Top-5 Accuracy', 0.60)*100:.1f}%", help="Correct career appears in top 5 predictions")

                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    fig_c1 = px.bar(comp_32, x='Model', y='Macro F1', color='Macro F1', title='Macro F1 Score (32 Classes)', text='Macro F1', color_continuous_scale='Viridis')
                    fig_c1.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                    st.plotly_chart(fig_c1, use_container_width=True)

                with col_b2:
                    fig_c2 = px.bar(comp_32, x='Model', y='Top-3 Acc', color='Top-3 Acc', title='Top-3 Ranking Accuracy (32 Classes)', text='Top-3 Acc', color_continuous_scale='Cividis')
                    fig_c2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                    st.plotly_chart(fig_c2, use_container_width=True)

    # ==========================================
    # 📖 ABOUT & RELIABILITY WARNINGS
    # ==========================================
    elif menu_option == "📖 About & Reliability Warnings":
        st.title("📖 System Architecture & Reliability Transparency")

        st.markdown("""
        <div class="warning-banner">
            <strong>⚠️ Model Reliability Warning:</strong> This system provides data-driven career recommendations based on the available dataset. It is intended as a decision-support tool and not as a definitive career predictor. Fine-grained predictions are limited by sample size per class.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📊 Scoring Methodology & Constants")
        st.markdown("""
        Level 2 Career Compatibility Scores are calculated using the following transparent weighted formula:
        $$\\text{Compatibility Score} = (0.45 \\times \\text{Skill Match \\%}) + (0.30 \\times \\text{Interest Match \\%}) + (0.15 \\times \\text{Education Match \\%}) + (0.10 \\times \\text{Domain ML Fit \\%})$$
        
        - **Skill Match (45%):** Proportion of candidate skills matching dataset career profiles.
        - **Interest Match (30%):** Proportion of candidate interests matching dataset career profiles.
        - **Education Match (15%):** Degree level compatibility based on dataset distributions.
        - **Domain ML Fit (10%):** Level 1 ML classifier confidence score.
        """)

        with st.expander("❓ Why is Top-3/Top-5 Accuracy evaluated?", expanded=True):
            st.markdown("""
            In career guidance, multiple career titles share overlapping skill requirements (e.g. *Software Developer*, *Software Engineer*, *Full Stack Developer*). Evaluating **Top-3 (55.0%)** and **Top-5 (60.0%)** accuracy provides a realistic measure of whether the system successfully includes the student's target field within its top recommendations.
            """)


if __name__ == '__main__':
    main()
