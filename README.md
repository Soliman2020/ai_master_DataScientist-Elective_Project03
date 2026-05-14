# StyleSense Recommendation Prediction

A Udacity Data Science elective project that builds a machine learning pipeline to predict whether customers recommend products based on their reviews.

## Project Summary

This project is for **StyleSense**, an online women's clothing retailer. The goal is to predict whether a customer recommends a product (`Recommended IND`: 1 = recommended, 0 = not recommended) based on their review data.

### Features

- **Numerical**: Age, Positive Feedback Count, Clothing ID
- **Categorical**: Division Name, Department Name, Class Name
- **Text**: Title, Review Text

### Pipeline Components

- **Text Processing**: spaCy for lemmatization, stop word removal, and tokenization
- **Feature Extraction**: TfidfVectorizer (5000 features, ngrams 1-2)
- **Preprocessing**: StandardScaler (numerical), OneHotEncoder (categorical)
- **Model**: Logistic Regression with GridSearchCV for hyperparameter tuning
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score

## Getting Started

### Dependencies

- scikit-learn
- pandas
- numpy
- matplotlib
- spacy
- jupyter

### Installation

1. Activate the virtual environment:

```bash
# Windows
.\venv\Scripts\Activate.ps1

# Mac/Linux
source venv/Scripts/activate
```

1. Install spaCy model:

```bash
python -m spacy download en_core_web_sm
```

1. Launch Jupyter notebook:

```bash
jupyter notebook starter/starter.ipynb
```

## Repository Files

| File/Directory             | Description                                |
| -------------------------- | ------------------------------------------ |
| `starter/starter.ipynb`    | Main Jupyter notebook with the ML pipeline |
| `starter/data/reviews.csv` | Dataset (18,442 customer reviews)          |
| `starter/models/recommendation_pipeline.pkl` | Saved model (pickle) |
| `README.md`                | This README file                           |
| `requirements.txt`         | Project dependencies                       |
| `LICENSE.txt`              | License agreement                          |
## Running the Project

1. Open `starter/starter.ipynb` in Jupyter
2. Run cells sequentially through:
   - Data loading and EDA
   - Pipeline building
   - Model training
   - Hyperparameter tuning with GridSearchCV

## Results

The tuned model achieves strong performance:

- **Accuracy**: 90.03%
- **Precision**: 92.70%
- **Recall**: 95.39%
- **F1 Score**: 94.03%

### Key Insights

**Top Predictive Features** (from text analysis):

- Positive: "love", "perfect", "great", "amazing", "comfortable"
- Negative: "disappointed", "return", "unflattering", "horrible"

Feature importance analysis shows that **text features** contribute most significantly to predictions, followed by categorical and numerical features.

## Model Deployment

The trained model is saved for future use:

- `models/recommendation_pipeline.pkl` - Serialized model (pickle)

### Using the Saved Model

```python
import pickle

# Load the model
with open('models/recommendation_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
prediction = model.predict(new_review_df)
probability = model.predict_proba(new_review_df)
```
