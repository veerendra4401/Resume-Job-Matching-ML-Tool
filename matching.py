# matching.py
import re
import json
import logging
from typing import List, Dict, Tuple, Set
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download required NLTK data (with error handling)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.warning("NLTK stopwords not found. Please run: nltk.download('stopwords')")

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.warning("NLTK wordnet not found. Please run: nltk.download('wordnet')")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.warning("NLTK punkt not found. Please run: nltk.download('punkt')")

_stopwords = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()

# Extended stopwords for better filtering
EXTENDED_STOPWORDS = _stopwords.union({
    'please', 'thank', 'thanks', 'hello', 'hi', 'dear', 'sincerely', 'regards',
    'looking', 'forward', 'opportunity', 'position', 'role', 'company', 'team',
    'work', 'job', 'resume', 'cv', 'description', 'requirement', 'skill'
})

def preprocess(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    """
    Advanced text preprocessing with configurable options.
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, emails, and special patterns
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\(\s*\)|\[\s*\]|\{\s*\}", " ", text)  # Remove empty brackets
    text = re.sub(r"[^\w\s]|_", " ", text)  # Keep words and spaces, remove punctuation
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    if not text:
        return ""
    
    # Tokenize
    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()
    
    # Filter tokens
    filtered_tokens = []
    for token in tokens:
        if len(token) <= 1:
            continue
        if remove_stopwords and token in EXTENDED_STOPWORDS:
            continue
        if token.isdigit():
            continue
            
        # Lemmatize if enabled
        if lemmatize:
            token = _lemmatizer.lemmatize(token)
        
        filtered_tokens.append(token)
    
    return " ".join(filtered_tokens)

def compute_similarity(resume_text: str, jd_text: str, method: str = "tfidf") -> float:
    """
    Compute similarity between resume and job description using multiple methods.
    
    Args:
        resume_text: Resume text content
        jd_text: Job description text content
        method: Similarity method ('tfidf', 'combined')
    
    Returns:
        Similarity score as percentage (0-100)
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0
    
    if method == "tfidf":
        return _compute_tfidf_similarity(resume_text, jd_text)
    elif method == "combined":
        return _compute_combined_similarity(resume_text, jd_text)
    else:
        return _compute_tfidf_similarity(resume_text, jd_text)

def _compute_tfidf_similarity(resume_text: str, jd_text: str) -> float:
    """TF-IDF based similarity calculation."""
    docs = [preprocess(resume_text), preprocess(jd_text)]
    
    if not docs[0].strip() or not docs[1].strip():
        return 0.0
    
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10000,
            min_df=1,
            max_df=0.9,
            stop_words=list(EXTENDED_STOPWORDS)
        )
        tfidf_matrix = vectorizer.fit_transform(docs)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(round(max(0, min(100, similarity * 100)), 2))
    except Exception as e:
        logger.error(f"Error in TF-IDF similarity calculation: {e}")
        return 0.0

def _compute_combined_similarity(resume_text: str, jd_text: str) -> float:
    """
    Combined similarity using TF-IDF and skill-based matching.
    """
    # TF-IDF similarity (70% weight)
    tfidf_sim = _compute_tfidf_similarity(resume_text, jd_text)
    
    # Skill-based similarity (30% weight)
    skills_list = load_skills()
    resume_skills = set(extract_skills_from_text(resume_text, skills_list))
    jd_skills = set(extract_skills_from_text(jd_text, skills_list))
    
    if jd_skills:
        skill_sim = (len(resume_skills & jd_skills) / len(jd_skills)) * 100
    else:
        skill_sim = 0.0
    
    combined_score = (tfidf_sim * 0.7) + (skill_sim * 0.3)
    return float(round(max(0, min(100, combined_score)), 2))

def load_skills(skills_file: str = "skills.txt") -> List[str]:
    """
    Load skills from file with enhanced error handling and categorization.
    """
    skills = []
    try:
        with open(skills_file, "r", encoding="utf-8") as f:
            for line in f:
                skill = line.strip()
                if skill and not skill.startswith("#"):  # Skip comments
                    skills.append(skill.lower())
        logger.info(f"Loaded {len(skills)} skills from {skills_file}")
    except FileNotFoundError:
        logger.warning(f"Skills file {skills_file} not found. Using default skills.")
        skills = get_default_skills()
    except Exception as e:
        logger.error(f"Error loading skills file: {e}")
        skills = get_default_skills()
    
    return sorted(list(set(skills)))

def get_default_skills() -> List[str]:
    """Comprehensive default skills list."""
    return [
        # Programming Languages
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "kotlin", "swift",
        "php", "ruby", "scala", "r", "matlab", "sql", "html", "css", "sass", "less",
        
        # Frameworks & Libraries
        "react", "angular", "vue", "node.js", "django", "flask", "spring", "express", "laravel",
        "ruby on rails", "asp.net", "tensorflow", "pytorch", "keras", "scikit-learn", "pandas",
        "numpy", "matplotlib", "d3.js", "jquery", "bootstrap", "tailwind",
        
        # Tools & Platforms
        "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git", "github", "gitlab",
        "jira", "confluence", "slack", "figma", "tableau", "power bi", "splunk",
        
        # Databases
        "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "oracle", "sql server",
        "cassandra", "dynamodb", "firebase",
        
        # Methodologies & Concepts
        "machine learning", "deep learning", "nlp", "computer vision", "data science",
        "devops", "agile", "scrum", "ci/cd", "microservices", "rest api", "graphql",
        "object oriented programming", "functional programming", "test driven development",
        
        # Soft Skills
        "leadership", "communication", "problem solving", "teamwork", "project management",
        "time management", "critical thinking", "creativity", "adaptability"
    ]

def extract_skills_from_text(text: str, skills_list: List[str]) -> List[str]:
    """
    Enhanced skill extraction with better matching and confidence scoring.
    """
    if not text or not skills_list:
        return []
    
    text_lower = text.lower()
    found_skills = []
    
    # Sort by length (longer first) to avoid partial matches
    skills_sorted = sorted(skills_list, key=len, reverse=True)
    
    for skill in skills_sorted:
        # Exact phrase matching with word boundaries
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_skills = []
    for skill in found_skills:
        if skill not in seen:
            seen.add(skill)
            unique_skills.append(skill)
    
    return unique_skills

def extract_skills_with_context(text: str, skills_list: List[str]) -> List[Dict]:
    """
    Extract skills with context and confidence scores.
    """
    skills_found = extract_skills_from_text(text, skills_list)
    result = []
    
    for skill in skills_found:
        # Find context around the skill
        pattern = r'\b' + re.escape(skill) + r'\b'
        matches = list(re.finditer(pattern, text.lower()))
        
        if matches:
            first_match = matches[0]
            start = max(0, first_match.start() - 50)
            end = min(len(text), first_match.end() + 50)
            context = text[start:end].strip()
            
            result.append({
                "skill": skill,
                "context": context,
                "count": len(matches),
                "confidence": min(1.0, len(matches) * 0.3)  # Simple confidence based on frequency
            })
    
    return sorted(result, key=lambda x: -x["confidence"])

def top_matching_terms(resume_text: str, jd_text: str, top_n: int = 15) -> List[Tuple[str, float]]:
    """
    Enhanced term matching with better filtering and scoring.
    """
    docs = [preprocess(resume_text), preprocess(jd_text)]
    
    if not docs[0].strip() or not docs[1].strip():
        return []
    
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=8000,
            stop_words=list(EXTENDED_STOPWORDS),
            min_df=1,
            max_df=0.85
        )
        
        tfidf_matrix = vectorizer.fit_transform(docs)
        jd_vector = tfidf_matrix[1].toarray()[0]
        feature_names = vectorizer.get_feature_names_out()
        
        # Get terms with highest TF-IDF in JD
        term_weights = [(feature_names[i], float(jd_vector[i])) 
                       for i in range(len(feature_names)) 
                       if jd_vector[i] > 0]
        
        # Sort by weight
        term_weights.sort(key=lambda x: -x[1])
        
        # Filter terms that appear in resume
        resume_processed = preprocess(resume_text)
        matched_terms = []
        
        for term, weight in term_weights:
            if term in resume_processed and len(term) > 2:
                matched_terms.append((term, weight))
                if len(matched_terms) >= top_n:
                    break
        
        return matched_terms
        
    except Exception as e:
        logger.error(f"Error in top matching terms: {e}")
        return []

def analyze_text_quality(text: str, text_type: str = "resume") -> Dict:
    """
    Analyze text quality and provide metrics.
    """
    words = preprocess(text).split()
    sentences = re.split(r'[.!?]+', text)
    
    metrics = {
        "word_count": len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "unique_words": len(set(words)) if words else 0,
        "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
        "vocabulary_richness": len(set(words)) / len(words) if words else 0
    }
    
    # Quality assessment
    if text_type == "resume":
        if metrics["word_count"] < 100:
            metrics["quality"] = "Too short - add more details"
        elif metrics["word_count"] > 1000:
            metrics["quality"] = "Good length"
        else:
            metrics["quality"] = "Adequate"
    else:  # JD
        if metrics["word_count"] < 50:
            metrics["quality"] = "Very brief - may lack detail"
        else:
            metrics["quality"] = "Detailed"
    
    return metrics

def generate_comprehensive_report(resume_text: str, jd_text: str, skills_file: str = "skills.txt") -> Dict:
    """
    Generate a comprehensive matching report with detailed analysis.
    """
    skills_list = load_skills(skills_file)
    
    # Basic extraction
    resume_skills = extract_skills_from_text(resume_text, skills_list)
    jd_skills = extract_skills_from_text(jd_text, skills_list)
    missing_skills = [s for s in jd_skills if s not in resume_skills]
    
    # Enhanced analysis
    resume_skills_detailed = extract_skills_with_context(resume_text, skills_list)
    jd_skills_detailed = extract_skills_with_context(jd_text, skills_list)
    
    # Multiple similarity scores
    tfidf_score = compute_similarity(resume_text, jd_text, method="tfidf")
    combined_score = compute_similarity(resume_text, jd_text, method="combined")
    
    # Top matching terms
    top_matches = top_matching_terms(resume_text, jd_text, top_n=15)
    
    # Text quality analysis
    resume_quality = analyze_text_quality(resume_text, "resume")
    jd_quality = analyze_text_quality(jd_text, "jd")
    
    # Skill matching metrics
    skill_match_ratio = len(set(resume_skills) & set(jd_skills)) / len(jd_skills) if jd_skills else 0
    skill_coverage = len(set(resume_skills) & set(jd_skills)) / len(set(jd_skills)) if jd_skills else 0
    
    report = {
        # Similarity Scores
        "similarity_score_percent": combined_score,
        "tfidf_score": tfidf_score,
        "combined_score": combined_score,
        
        # Skills Analysis
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
        "missing_skills_from_jd": missing_skills,
        "resume_skills_detailed": resume_skills_detailed,
        "jd_skills_detailed": jd_skills_detailed,
        
        # Matching Details
        "top_matching_terms": top_matches,
        "skill_match_ratio": round(skill_match_ratio * 100, 2),
        "skill_coverage_percent": round(skill_coverage * 100, 2),
        
        # Text Quality
        "resume_quality": resume_quality,
        "jd_quality": jd_quality,
        
        # Statistics
        "skill_statistics": {
            "resume_skill_count": len(resume_skills),
            "jd_skill_count": len(jd_skills),
            "common_skills_count": len(set(resume_skills) & set(jd_skills)),
            "missing_skills_count": len(missing_skills)
        },
        
        # Timestamp
        "analysis_timestamp": np.datetime64('now').astype(str)
    }
    
    return report

def generate_report(resume_text: str, jd_text: str, skills_file: str = "skills.txt") -> Dict:
    """
    Main report generation function (maintains backward compatibility).
    """
    return generate_comprehensive_report(resume_text, jd_text, skills_file)

def save_report_json(report: Dict, out_path: str = "report.json"):
    """Save report to JSON file with error handling."""
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved to {out_path}")
    except Exception as e:
        logger.error(f"Error saving report: {e}")
        raise

def get_improvement_suggestions(report: Dict) -> List[str]:
    """Generate improvement suggestions based on the report."""
    suggestions = []
    score = report["similarity_score_percent"]
    missing_skills = report["missing_skills_from_jd"]
    skill_stats = report["skill_statistics"]
    
    # Score-based suggestions
    if score < 40:
        suggestions.extend([
            "🔴 **Major improvements needed**: Focus on better alignment with job requirements",
            "• Add more keywords from the job description throughout your resume",
            "• Highlight projects and experiences that match the job responsibilities",
            "• Consider gaining experience in missing required skills"
        ])
    elif score < 70:
        suggestions.extend([
            "🟡 **Good foundation, needs optimization**: Refine your resume for better matching",
            "• Reorder content to highlight most relevant skills and experiences first",
            "• Add specific achievements and metrics related to required skills",
            "• Include a skills summary section with key technologies"
        ])
    else:
        suggestions.extend([
            "🟢 **Strong match**: Focus on presentation and interview preparation",
            "• Ensure your best matching skills are prominently featured",
            "• Prepare detailed examples of how you've used key technologies",
            "• Quantify your achievements with specific numbers and results"
        ])
    
    # Skill-based suggestions
    if missing_skills:
        suggestions.append(f"• **Priority skills to acquire**: {', '.join(missing_skills[:5])}")
    
    if skill_stats['resume_skill_count'] < 8:
        suggestions.append("• **Expand your technical skills section** with more technologies")
    
    if report['resume_quality']['word_count'] < 150:
        suggestions.append("• **Add more detail** to your resume - include specific projects and achievements")
    
    # General suggestions
    suggestions.extend([
        "• **Use action verbs** to start bullet points (e.g., 'Developed', 'Managed', 'Improved')",
        "• **Include metrics** wherever possible (e.g., 'improved performance by 25%')",
        "• **Tailor your summary** to match the specific job requirements",
        "• **Proofread carefully** for spelling and grammar errors"
    ])
    
    return suggestions
def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts (wrapper for compute_similarity).
    This function is maintained for backward compatibility with the Streamlit app.
    """
    return compute_similarity(text1, text2)