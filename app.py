# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import base64
from matching import generate_report, save_report_json, extract_skills_from_text

# Page configuration
st.set_page_config(
    page_title="Resume ↔ Job Match Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .skill-match {
        background-color: #d4edda;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem;
    }
    .skill-missing {
        background-color: #f8d7da;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem;
    }
    .suggestion-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

def create_radar_chart(report):
    """Create radar chart for skills comparison"""
    categories = ['Technical Skills', 'Tools & Technologies', 'Experience Match', 'Education', 'Overall Fit']
    
    # Calculate scores based on actual report data
    skill_ratio = report.get('skill_match_ratio', 0)
    similarity_score = report['similarity_score_percent']
    
    resume_scores = [
        skill_ratio,  # Technical Skills
        min(skill_ratio * 1.2, 100),  # Tools & Technologies
        similarity_score * 0.9,  # Experience Match
        min(similarity_score * 1.1, 100),  # Education
        similarity_score  # Overall Fit
    ]
    
    job_scores = [100, 100, 100, 100, 100]  # Ideal scores
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=resume_scores,
        theta=categories,
        fill='toself',
        name='Your Resume',
        line_color='#1f77b4'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=job_scores,
        theta=categories,
        fill='toself',
        name='Job Requirements',
        line_color='#ff7f0e',
        opacity=0.3
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Skills Radar Chart - Resume vs Job Requirements"
    )
    
    return fig

def create_score_gauge(score):
    """Create a gauge chart for the main score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Match Score"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "lightblue"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def get_color_for_score(score):
    """Return color based on score"""
    if score >= 70:
        return "green"
    elif score >= 40:
        return "orange"
    else:
        return "red"

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")
    
    analysis_mode = st.selectbox(
        "Analysis Mode",
        ["Standard", "Detailed", "Fresher Focus", "Experienced Professional"]
    )
    
    skill_threshold = st.slider(
        "Skill Matching Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        help="Adjust how strictly skills are matched"
    )
    
    show_technical_details = st.checkbox("Show Technical Details", value=False)
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.info("""
    - Paste clean, formatted text for best results
    - Include specific skills and technologies
    - Remove personal information before uploading
    - Use detailed job descriptions for better matching
    """)

# Main content
st.markdown('<div class="main-header">📊 Resume ↔ Job Description Match Analyzer</div>', unsafe_allow_html=True)

# File upload and text input section
col1, col2 = st.columns(2)

with col1:
    st.header("📄 Resume Input")
    
    input_method_resume = st.radio(
        "Resume Input Method",
        ["Paste Text", "Upload File"],
        horizontal=True,
        key="resume_input"
    )
    
    if input_method_resume == "Paste Text":
        resume_text = st.text_area(
            "Paste your resume text here:",
            height=300,
            placeholder="Paste your resume content here...\n\nInclude:\n- Skills\n- Experience\n- Education\n- Projects\n- Certifications"
        )
    else:
        uploaded_resume = st.file_uploader(
            "Upload Resume",
            type=["txt"],
            key="resume_upload"
        )
        if uploaded_resume is not None:
            try:
                content = uploaded_resume.read().decode("utf-8")
                resume_text = content
                st.success(f"✅ Resume loaded successfully! ({len(resume_text)} characters)")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                resume_text = ""
        else:
            resume_text = ""

with col2:
    st.header("💼 Job Description Input")
    
    input_method_jd = st.radio(
        "Job Description Input Method",
        ["Paste Text", "Upload File"],
        horizontal=True,
        key="jd_input"
    )
    
    if input_method_jd == "Paste Text":
        jd_text = st.text_area(
            "Paste job description here:",
            height=300,
            placeholder="Paste the job description here...\n\nLook for:\n- Required skills\n- Technologies\n- Qualifications\n- Responsibilities"
        )
    else:
        uploaded_jd = st.file_uploader(
            "Upload Job Description",
            type=["txt"],
            key="jd_upload"
        )
        if uploaded_jd is not None:
            try:
                content_jd = uploaded_jd.read().decode("utf-8")
                jd_text = content_jd
                st.success(f"✅ Job description loaded successfully! ({len(jd_text)} characters)")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                jd_text = ""
        else:
            jd_text = ""

# Quick stats
if resume_text and jd_text:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Resume Length", f"{len(resume_text.split())} words")
    with col2:
        st.metric("JD Length", f"{len(jd_text.split())} words")
    with col3:
        resume_skills_preview = extract_skills_from_text(resume_text, "skills.txt")
        st.metric("Skills Detected", len(resume_skills_preview))
    with col4:
        jd_skills_preview = extract_skills_from_text(jd_text, "skills.txt")
        st.metric("JD Requirements", len(jd_skills_preview))

st.markdown("---")

# Analysis button and results
analysis_col1, analysis_col2 = st.columns([1, 4])

with analysis_col1:
    analyze_btn = st.button(
        "🚀 Analyze Match",
        type="primary",
        use_container_width=True
    )

if analyze_btn:
    if not resume_text or not jd_text:
        st.error("❌ Please provide both resume and job description content.")
    else:
        with st.spinner("🔍 Analyzing your resume against job requirements..."):
            # Generate comprehensive report
            report = generate_report(resume_text, jd_text, skills_file="skills.txt")
            
            # Display main score with gauge
            score = report["similarity_score_percent"]
            
            st.markdown("## 📈 Match Analysis Results")
            
            # Main score gauge
            gauge_col, stats_col = st.columns([2, 1])
            
            with gauge_col:
                fig = create_score_gauge(score)
                st.plotly_chart(fig, use_container_width=True)
            
            with stats_col:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Overall Match", f"{score}%")
                st.metric("Skills Match", f"{report['skill_statistics']['common_skills_count']}/{report['skill_statistics']['jd_skill_count']}")
                st.metric("Missing Skills", report['skill_statistics']['missing_skills_count'])
                st.metric("Matching Terms", len(report["top_matching_terms"]))
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Radar chart
            st.plotly_chart(create_radar_chart(report), use_container_width=True)
            
            # Skills analysis in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["🎯 Skills Analysis", "📊 Matching Details", "💡 Suggestions", "📋 Full Report"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("✅ Your Skills")
                    if report["resume_skills"]:
                        for skill in report["resume_skills"]:
                            st.markdown(f'<div class="skill-match">{skill}</div>', unsafe_allow_html=True)
                    else:
                        st.info("No skills detected from our skills list.")
                
                with col2:
                    st.subheader("🎯 Required Skills")
                    if report["jd_skills"]:
                        for skill in report["jd_skills"]:
                            if skill in report["missing_skills_from_jd"]:
                                st.markdown(f'<div class="skill-missing">{skill} ❌</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="skill-match">{skill} ✅</div>', unsafe_allow_html=True)
                    else:
                        st.info("No specific skills detected in job description.")
                
                # Missing skills emphasis
                if report["missing_skills_from_jd"]:
                    st.error(f"**Missing Skills:** {', '.join(report['missing_skills_from_jd'])}")
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Top Matching Terms")
                    if report["top_matching_terms"]:
                        for term, weight in report["top_matching_terms"][:10]:
                            st.write(f"• {term} (weight: {weight:.3f})")
                    else:
                        st.info("No significant matching terms found.")
                
                with col2:
                    st.subheader("Analysis Details")
                    st.write(f"**Similarity Algorithm:** TF-IDF + Cosine Similarity")
                    st.write(f"**Analysis Mode:** {analysis_mode}")
                    st.write(f"**Skill Match Ratio:** {report.get('skill_match_ratio', 0)}%")
                    
                    if show_technical_details:
                        st.json({
                            "tfidf_score": report.get("tfidf_score", 0),
                            "combined_score": report.get("combined_score", 0),
                            "skill_counts": report.get("skill_statistics", {})
                        })
            
            with tab3:
                st.markdown('<div class="suggestion-box">', unsafe_allow_html=True)
                st.subheader("💡 Resume Improvement Suggestions")
                
                suggestions = []
                
                # Score-based suggestions
                if score < 40:
                    suggestions.extend([
                        "🔴 **Low Match Score**: Consider major revisions to better align with the job requirements",
                        "• Add more relevant keywords from the job description",
                        "• Highlight transferable skills that match the requirements",
                        "• Include specific technologies and tools mentioned in the JD"
                    ])
                elif score < 70:
                    suggestions.extend([
                        "🟡 **Moderate Match**: Good foundation but needs optimization",
                        "• Emphasize your most relevant projects and experiences",
                        "• Add missing skills through online courses or projects",
                        "• Reorder content to highlight most relevant qualifications first"
                    ])
                else:
                    suggestions.extend([
                        "🟢 **Strong Match**: Your resume aligns well with the job",
                        "• Ensure your best matching skills are prominently featured",
                        "• Quantify achievements related to required skills",
                        "• Prepare to discuss these key areas in interviews"
                    ])
                
                # Skill-based suggestions
                if report["missing_skills_from_jd"]:
                    suggestions.append(f"• **Focus on acquiring/displaying**: {', '.join(report['missing_skills_from_jd'][:5])}")
                
                if report['skill_statistics']['resume_skill_count'] < 5:
                    suggestions.append("• **Add more technical skills** to your resume")
                
                # Formatting suggestions
                suggestions.extend([
                    "• Use bullet points for better readability",
                    "• Include metrics and achievements (e.g., 'improved efficiency by 20%')",
                    "• Keep resume length appropriate (1-2 pages for most roles)"
                ])
                
                for suggestion in suggestions:
                    st.write(suggestion)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab4:
                st.subheader("Complete Analysis Report")
                st.json(report)
                
                # Download options
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("💾 Save Detailed Report", use_container_width=True):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"resume_analysis_report_{timestamp}.json"
                        save_report_json(report, out_path=filename)
                        st.success(f"Report saved as {filename}")
                
                with col2:
                    # Create downloadable JSON
                    json_report = json.dumps(report, indent=2)
                    b64 = base64.b64encode(json_report.encode()).decode()
                    href = f'<a href="data:application/json;base64,{b64}" download="resume_analysis_report.json">📥 Download JSON Report</a>'
                    st.markdown(href, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Resume ↔ Job Match Analyzer | Uses TF-IDF + Cosine Similarity | Good for freshers' portfolios</p>
    </div>
    """,
    unsafe_allow_html=True
)