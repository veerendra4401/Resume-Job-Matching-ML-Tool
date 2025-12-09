# 🚀 Resume ↔ Job Description Matching Tool  
### **AI-powered ML/NLP app that analyzes how well a resume matches a job description**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Machine Learning](https://img.shields.io/badge/ML-TF--IDF%20%7C%20Cosine%20Similarity-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)

---

### 📌 **Overview**  
This project is a simple yet powerful **Machine Learning + NLP-based tool** that measures how well a **resume matches a job description**.  
It helps job seekers improve their resumes and helps recruiters identify suitable candidates faster.

---

# ✨ Features

### 🔍 **1. Match Score (0–100%)**
Calculates the similarity between the resume and job description using:  
- **TF-IDF Vectorization**  
- **Cosine Similarity**

### 🧠 **2. Automatic Skill Extraction**
Finds skills mentioned in:
- Resume  
- Job Description  

Uses a customizable `skills.txt` file.

### ⚠️ **3. Missing Skills Detection**
Shows important skills from JD that are **not present** in the resume.

### 💡 **4. Improvement Suggestions**
AI-based suggestions to improve resume match score.

### 📝 **5. JSON Report Export**
Download a complete structured report.

### 🌐 **6. Streamlit Web App**
User-friendly UI with file upload + text paste support.

---

# 🧱 Architecture

```
Resume Text / File       Job Description Text / File
        │                           │
        └──────────────┬────────────┘
                       ▼
               Preprocessing (NLP)
                       |
              TF-IDF Vectorizer
                       |
                Cosine Similarity
                       |
         Skills Extractor (skills.txt)
                       |
         Match Score + Missing Skills
                       |
                  Streamlit UI
```

---

# 📂 **Project Structure**
```
resume-job-matcher/
│
├── app.py                 # Streamlit frontend
├── matching.py            # ML + NLP model logic
├── skills.txt             # Skills dictionary
├── requirements.txt       # Dependencies
├── README.md              # (this file)
│
└── sample_data/
    ├── sample_resume.txt
    └── sample_job_description.txt
```

---

# ⚙️ Installation & Setup

### **1. Clone the repository**
```bash
git clone https://github.com/your-username/resume-job-matcher.git
cd resume-job-matcher
```

### **2. Create a virtual environment**
```bash
python -m venv venv
```

Activate it:  
- Windows → `venv\Scripts\activate`  
- Mac/Linux → `source venv/bin/activate`

### **3. Install dependencies**
```bash
pip install -r requirements.txt
```

### **4. Download NLTK data**
```bash
python -m nltk.downloader punkt stopwords wordnet
```

### **5. Run the Streamlit app**
```bash
streamlit run app.py
```

---

# 🎯 Usage

### ✔ Paste or upload:
- Resume text (left pane)  
- Job description (right pane)

### ✔ Click **Analyze Match**  
You will get:
- Match Score  
- Skills found  
- Missing skills  
- Top matching keywords  
- Suggestions to improve your resume  

---

# 📊 Sample Output

| Metric | Result |
|--------|--------|
| **Match Score** | 72% |
| **Resume Skills** | Python, Pandas, NumPy |
| **JD Skills** | Python, ML, Scikit-learn |
| **Missing Skills** | ML, Scikit-learn |

---

# 🌟 Why This Project is Impressive for Freshers?

✔ Shows real ML + NLP skills  
✔ Recruiters love resume-related tools  
✔ End-to-end project (data → model → UI → output)  
✔ Clean, understandable Python code  
✔ Easy to deploy on Streamlit Cloud or Render  
✔ Looks amazing on GitHub & CV  

---

# 🚀 Future Enhancements (Roadmap)

- [ ] Support **PDF** and **DOCX** parsing  
- [ ] Add **semantic embeddings** using SentenceTransformers  
- [ ] Build **ranking engine** to compare multiple resumes  
- [ ] Add **dashboard** for HR analytics  
- [ ] Add **Flask API backend**  
- [ ] Add **Dockerfile** for deployment  

---

# 🛠 Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python |
| ML | TF-IDF, Cosine Similarity |
| NLP | NLTK |
| UI | Streamlit |
| Data Formats | TXT, JSON |

---

# 🤝 Contributing  
PRs are welcome!  
Feel free to fork, open issues, or submit improvements.

---

# 📄 License  
MIT License — free to use and modify.

---

# ⭐ If you like this project, give it a star on GitHub!
Your support motivates creation of more beginner-friendly ML/NLP tools.  

