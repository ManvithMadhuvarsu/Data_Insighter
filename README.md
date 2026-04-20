# Data Insighter 📊

**Data Insighter** is a professional-grade, full-stack data analysis and visualization platform. It empowers users to transform raw datasets into actionable insights through an intuitive web interface, advanced statistical processing, and high-fidelity interactive dashboards.

---

## 🚀 Key Features

- **Multi-Format Data Support**: Seamlessly upload and process `.csv` and `.json` files.
- **Automated Data Profiling**: Instantly view column summaries, null counts, unique values, and data types upon upload.
- **Interactive Visualization Engine**: Generate dynamic charts (Bar, Line, Scatter, Heatmaps, etc.) using Plotly, Chart.js, and Bokeh.
- **Advanced Analytics**: Built-in support for statistical analysis, survival analysis, and Bayesian modeling.
- **Secure Authentication**: Robust user management system with hashed passwords and CSRF protection.
- **High-Fidelity Exports**: Export individual visualizations or entire dashboards as high-resolution images or interactive HTML files.
- **Sample Datasets**: Built-in sample datasets (e.g., Global Superstore) for immediate exploration.

---

## 🛠️ Tech Stack

### Backend
- **Framework**: Flask (Python)
- **Data Processing**: Pandas, NumPy, Scipy
- **Machine Learning & Stats**: Scikit-learn, Statsmodels, Pingouin, Lifelines
- **Security**: Werkzeug (Hashing), Dotenv, Flask-Session, CSRF protection

### Frontend
- **Structure**: Semantic HTML5
- **Styling**: Modern, responsive Vanilla CSS
- **Interactivity**: JavaScript (ES6+), Chart.js
- **Visualization**: Plotly, Bokeh, Altair

### Infrastructure
- **Version Control**: Git / GitHub
- **Deployment Ready**: Gunicorn / Procfile included
- **Background Tasks**: Celery & Redis (support built-in)

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8+
- Git

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/ManvithMadhuvarsu/Data_Insighter.git
   cd Data_Insighter
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   SECRET_KEY=your_super_secret_key_here
   ```

5. **Run the application**:
   ```bash
   python app.py
   ```
   Access the app at `http://127.0.0.1:5000`

---

## 🔒 Security & Recent Updates

Recent updates have focused on hardening the application for production use:
- **CSRF Protection**: All forms and API requests are now protected against Cross-Site Request Forgery.
- **Secure Authentication**: Switched to PBKDF2 hashing for user passwords.
- **Session Management**: Server-side session handling to prevent sensitive data leakage.
- **Path Validation**: Strict validation for file paths to prevent directory traversal attacks.
- **Dashboard Optimization**: Improved scaling and text legibility for exported visualizations.

---

## 📁 Project Structure

```text
Data_Insighter/
├── app.py                # Main Flask entry point
├── data_processor.py      # Core data manipulation logic
├── visualization_generator.py # Chart & Dashboard logic
├── file_utils.py          # File I/O utilities
├── uploads/               # Temporary user data (volatile)
├── sample_datasets/       # Included datasets for testing
├── static/                # Modern CSS and JS assets
├── templates/             # HTML Jinja2 templates
└── requirements.txt       # Project dependencies
```

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details (if available).

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for any bugs or feature requests.

---

*Developed with ❤️ by [Manvith Madhuvarsu](https://github.com/ManvithMadhuvarsu)*
