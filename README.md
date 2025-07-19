# Credit-Card-Fraud-Detection-System
## 🔍 Overview
An intelligent machine learning system for detecting fraudulent credit card transactions using multiple ML algorithms and ensemble methods. The system provides a web-based dashboard for real-time fraud detection, data visualization, and model performance analysis.

## ✨ Features
- **10 Machine Learning Models**: Logistic Regression, SVM, KNN, Random Forest, Decision Tree, Gradient Boosting, XGBoost, AdaBoost, Balanced Random Forest, Easy Ensemble
- **Real-time Fraud Detection**: Interactive web interface with model selection
- **Data Visualizations**: Class distribution, transaction amounts, feature analysis with Plotly.js
- **Model Performance Metrics**: Accuracy scores and probability predictions
- **Imbalanced Data Handling**: Specialized ensemble methods for fraud detection
- **Responsive Design**: Mobile-friendly dashboard

## 🛠️ Tech Stack
- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **Machine Learning**: scikit-learn, XGBoost, imbalanced-learn
- **Data Visualization**: Plotly.js
- **Data Processing**: pandas, numpy

## 📁 Project Structure
```markdown
.
├── app.py                  # Main Flask application
├── requirements.txt         # Python dependencies
├── templates/               # HTML templates
│   ├── layout.html         # Base layout
│   ├── index.html          # Home page
│   ├── dashboard.html       # Dashboard page
│   └── results.html         # Results page
├── static/                 # Static files (CSS, JavaScript, images)
│   ├── css/
│   ├── js/
│   └── images/
├── data/                   # Data files
│   ├── transactions.csv     # Sample transaction data
│   └── models/             # Trained machine learning models
├── logs/                   # Log files
└── README.md               # Project documentation
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start
1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/Intelligent-Fraud-Prevention-in-Credit-Card-Transactions.git
cd Intelligent-Fraud-Prevention-in-Credit-Card-Transactions
```
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py`
4. Access the dashboard at `http://127.0.0.1:5000`

## 🖥️ Usage
Web Dashboard Navigation
🏠 Dashboard: Upload CSV data and view transaction overview
📊 Visualizations: Interactive charts and graphs
📈 Analysis: Detailed transaction amount analysis
🤖 ML Model: Real-time fraud prediction interface
📚 Theory: Background on fraud detection techniques
🔍 Features: Feature importance analysis
📉 Amount Trends: Transaction amount trend analysis


## 🤖 Machine Learning Models
logreg - Logistic Regression
svm - Support Vector Machine
knn - K-Nearest Neighbors
rf - Random Forest
dt - Decision Tree
gb - Gradient Boosting
xgb - XGBoost
adaboost - AdaBoost
brf - Balanced Random Forest
easy_ensemble - Easy Ensemble


## 🎯 Performance Metrics
Accuracy: Overall prediction accuracy (typically >99%)
Precision: Fraud detection precision
Recall: Fraud detection rate
F1-Score: Balanced metric for imbalanced data
Response Time: <500ms per prediction


## 📋 Dependencies

````markdown
Flask>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.2.0
xgboost>=1.7.0
imbalanced-learn>=0.10.0
````

## 🔧 Configuration
Model Directory: ml model/
Template Directory: templates
Static Files: static
Host: 0.0.0.0 (production) / localhost (development)
Port: Environment variable PORT or default 5000

## 🤝 Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a new branch: `git checkout -b feature/YourFeature`
3. Make your changes and commit them: `git commit -m 'Add new feature'`
4. Push to the branch: `git push origin feature/YourFeature`
5. Create a pull request

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Md. Shakil Hossain**
- GitHub: [@shakiliitju](https://github.com/shakiliitju)
- Project: [Credit-Card-Fraud-Detection-System](https://github.com/shakiliitju/Credit-Card-Fraud-Detection-System)


## 👥 Acknowledgments
- [Flask](https://flask.palletsprojects.com/) - The web framework used
- [scikit-learn](https://scikit-learn.org/) - Machine learning library
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting library
- [Plotly.js](https://plotly.com/javascript/) - Data visualization library
- [imbalanced-learn](https://imbalanced-learn.org/) - Imbalanced data handling

