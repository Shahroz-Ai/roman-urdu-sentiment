# 💬 Roman Urdu Sentiment Analyzer

> Classifies Roman Urdu text (Urdu written in English letters) as **Positive**, **Neutral**, or **Negative**. 😊😐😠

🔗 **Live demo:** [Try it here](https://roman-urdu-sentiment-shahroz-ai.streamlit.app/)
## 🎯 What is this?

Roman Urdu has no standard spelling ("acha", "achha", "achaa" are all common), and
very little ready-made NLP tooling exists for it. This project trains and evaluates
a sentiment classifier for it, then serves it in a small web app.

## 📊 Results

Evaluated on a held-out test set of 3,875 texts (20% of the data, split before any model selection):

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Always predict "Neutral" (dumb baseline) | 0.431 | 0.201 |
| Word TF-IDF + Logistic Regression | 0.655 | 0.652 |
| **Word + character TF-IDF + Logistic Regression (final)** | **0.678** | **0.672** |

Models were compared with 5-fold cross-validation on the training data only
(macro F1): word TF-IDF + Naive Bayes 0.608, word TF-IDF + LogReg 0.625,
word+char + LinearSVC 0.635, char n-grams + LogReg 0.647, **word+char + LogReg 0.656**.
The test set was used once, for the final score.

## 🔍 What I learned

- 🔤 **Character n-grams help**: splitting words into 2-5 letter pieces handles
  spelling variations (a misspelled "bakwaz" is still recognized as "bakwaas")
- 🧹 **Data cleaning mattered**: about 700 duplicate or contradictory texts were removed
  before splitting, so the same sentence cannot appear in both train and test
- 📏 **Macro F1 over accuracy**: Neutral is 43% of the data, so accuracy alone is misleading
- 🏷️ **Labels are noisy**: some confident "mistakes" looked like label errors (for example,
  clear praise labeled Neutral), which limits the achievable score
- 🎚️ **Confidence is meaningful**: on the test set, predictions with confidence above 0.75
  were about 87% correct, while those below 0.45 were only about 41% correct
- 🔀 **Most errors involve Neutral**: about 76% of mistakes confuse Neutral with Positive
  or Negative; Positive vs Negative is rarely mixed up

## ⚠️ Limitations

- 🧠 Bag-of-words model: it does not understand context or negation (for example, an
  article saying a team *won* may be read as positive for everyone)
- 🔀 Mixed sentences ("good phone but bad battery") get split probabilities
- 🏷️ Dataset labels are noisy, so treat results as approximate
- 📚 Trained on one dataset; it may not transfer to other domains

## 🛠️ Tech stack

Python, pandas, scikit-learn, Streamlit

## 📁 Project structure

```
├── prepare_data.py     # cleans text, removes duplicates and conflicting labels
├── train_baseline.py   # dumb baseline and first model
├── train_compare.py    # cross-validated model comparison, saves model.pkl
├── error_analysis.py   # confidence check, confident mistakes, top words
├── app.py              # Streamlit app
├── model.pkl           # trained model
└── requirements.txt
```

## 🚀 Run locally

1. `pip install -r requirements.txt`
2. Download the dataset (see below) and run `python prepare_data.py`
3. Train with `python train_compare.py`
4. Start the app: `python -m streamlit run app.py`

## 📚 Dataset

Roman Urdu Data Set by Zareen Sharf (2017), UCI Machine Learning Repository,
https://doi.org/10.24432/C58325. About 20,000 texts labeled Positive, Negative,
or Neutral. The dataset is not included in this repository; please download it
from the UCI page and credit the original author.

---
Made with ❤️ by **Shahroz**
