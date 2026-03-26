import { useState } from "react";

const steps = [
  {
    num: 1,
    title: "Libraries Import & Data Load",
    emoji: "📦",
    tag: "Setup",
    tagColor: "bg-slate-100 text-slate-600",
    code: `import numpy as np
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
import lightgbm as lgb
import scipy.sparse as sp

train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
sub   = pd.read_csv('Sample.csv')`,
    points: [
      { label: "numpy / pandas", desc: "Math calculations + table data handle pannrom" },
      { label: "re", desc: "Regex — text patterns find & replace pannrom" },
      { label: "warnings.filterwarnings('ignore')", desc: "Unnecessary warning messages hide pannrom" },
      { label: "TfidfVectorizer", desc: "Comment text → numbers convert pannrom" },
      { label: "LabelEncoder", desc: "'Hindu','Muslim' → 1,2,3 nu convert pannrom" },
      { label: "StratifiedKFold", desc: "Data-ai balanced-aa 5 parts-aa split pannrom" },
      { label: "lightgbm", desc: "Main ML model — fast & powerful" },
      { label: "scipy.sparse", desc: "80,000 column matrix-ai memory efficient-aa store pannrom" },
      { label: "3 CSV files", desc: "train (labels irukkum) + test (predict pannanum) + Sample (submission format)" },
    ]
  },
  {
    num: 2,
    title: "Train + Test Combine",
    emoji: "🔗",
    tag: "Data Prep",
    tagColor: "bg-blue-50 text-blue-600",
    code: `train['is_train'] = 1
test['is_train']  = 0
test['label']     = -1

df = pd.concat([train, test], axis=0, ignore_index=True)`,
    points: [
      { label: "is_train = 1 / 0", desc: "Flag vachu later train vs test distinguish pannrom" },
      { label: "test['label'] = -1", desc: "Test-la labels illai — placeholder -1 kudukrom" },
      { label: "pd.concat", desc: "Train + Test rows combine pannrom (row-wise stack)" },
      { label: "Yen combine pannrom?", desc: "TF-IDF fit pannum pothu both train+test words learn aaganum — illa na test-la new words miss aagum!" },
    ]
  },
  {
    num: 3,
    title: "Text Cleaning",
    emoji: "🧹",
    tag: "NLP",
    tagColor: "bg-green-50 text-green-600",
    code: `def clean_text(text):
    if pd.isna(text): return ''
    text = str(text)
    text = re.sub(r'http\\S+|www\\.\\S+', ' URL ', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

df['comment_clean'] = df['comment'].apply(clean_text)`,
    points: [
      { label: "pd.isna(text)", desc: "Null/empty check — null iruntha empty string return pannrom" },
      { label: "str(text)", desc: "Numbers or other types iruntha string-aa convert" },
      { label: "re.sub(r'http\\S+'...)", desc: "URLs ellam → ' URL ' replace pannrom. Model URL text learn pannaama irukka" },
      { label: "re.sub(r'\\s+',' ',...)", desc: "Multiple spaces → single space. 'hello   world' → 'hello world'" },
      { label: ".strip()", desc: "Starting/ending spaces remove pannrom" },
      { label: ".apply(clean_text)", desc: "Every row-ku inda function run pannrom" },
    ]
  },
  {
    num: 4,
    title: "Time Features",
    emoji: "⏰",
    tag: "Feature Eng",
    tagColor: "bg-purple-50 text-purple-600",
    code: `df['created_date'] = pd.to_datetime(df['created_date'], utc=True, errors='coerce')
df['hour']       = df['created_date'].dt.hour
df['dayofweek']  = df['created_date'].dt.dayofweek
df['month']      = df['created_date'].dt.month
df['is_night']   = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)`,
    points: [
      { label: "pd.to_datetime", desc: "String '2023-01-15 22:30' → datetime object convert pannrom" },
      { label: "errors='coerce'", desc: "Invalid date iruntha NaT (null) aagum, error throw aagaadu" },
      { label: "dt.hour / dayofweek / month", desc: "Date object-la irunthu hour, day, month extract pannrom" },
      { label: "is_night", desc: "10pm to 5am → 1, else 0. Rathiri comments toxic more aagum!" },
      { label: "is_weekend", desc: "dayofweek >= 5 means Saturday/Sunday → 1" },
      { label: "Yen time features?", desc: "Comment posted time-um category-um correlate aagalaam — pattern irukkum!" },
    ]
  },
  {
    num: 5,
    title: "Text Stat Features",
    emoji: "📝",
    tag: "Feature Eng",
    tagColor: "bg-purple-50 text-purple-600",
    code: `df['comment_len']  = df['comment_clean'].str.len()
df['word_count']   = df['comment_clean'].str.split().str.len()
df['avg_word_len'] = df['comment_len'] / (df['word_count'] + 1)
df['has_url']      = df['comment_clean'].str.contains('URL').astype(int)
df['exclaim_count']  = df['comment'].str.count('!')
df['question_count'] = df['comment'].str.count(r'\\?')
df['caps_ratio'] = df['comment'].apply(
    lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1))`,
    points: [
      { label: "comment_len", desc: "Total character count. Long comments = detailed/opinion type" },
      { label: "word_count", desc: "str.split() → list of words → len(). Word count." },
      { label: "avg_word_len", desc: "comment_len / word_count. Big words = formal, short = casual" },
      { label: "+1 in denominator", desc: "word_count = 0 iruntha division by zero avoid pannrom!" },
      { label: "exclaim_count", desc: "'!!!' = excited or angry comment signal" },
      { label: "caps_ratio", desc: "ALL CAPS = shouting/angry. 'HATE THIS' → high caps ratio" },
    ]
  },
  {
    num: 6,
    title: "Identity Features",
    emoji: "🏷️",
    tag: "Feature Eng",
    tagColor: "bg-purple-50 text-purple-600",
    code: `for col in ['race', 'religion', 'gender']:
    df[col] = df[col].fillna('not_detected')

le = LabelEncoder()
for col in ['race', 'religion', 'gender']:
    df[col + '_enc'] = le.fit_transform(df[col])

df['any_identity'] = (
    (df['race'] != 'not_detected') |
    (df['religion'] != 'not_detected') |
    (df['gender'] != 'not_detected')
).astype(int)`,
    points: [
      { label: "fillna('not_detected')", desc: "Null = 'not detected' nu treat pannrom. Null-aa vitaa LabelEncoder fail aagum" },
      { label: "LabelEncoder", desc: "'Hindu'→1, 'Muslim'→2, 'Christian'→3. Text → number" },
      { label: "fit_transform", desc: "fit = learn unique values, transform = convert to numbers. Both oru step-la" },
      { label: "any_identity", desc: "Race OR religion OR gender mentioned → 1 else 0. Binary flag" },
      { label: "identity_count", desc: "How many identities mentioned? 0,1,2 or 3. More = sensitive comment" },
    ]
  },
  {
    num: 7,
    title: "Votes & Interactions",
    emoji: "🗳️",
    tag: "Feature Eng",
    tagColor: "bg-purple-50 text-purple-600",
    code: `df['if1_x_if2']  = df['if_1'] * df['if_2']
df['if2_is_4']   = (df['if_2'] == 4).astype(int)
df['vote_diff']  = df['upvote'] - df['downvote']
df['vote_ratio'] = df['upvote'] / (df['downvote'] + 1)
df['total_votes']= df['upvote'] + df['downvote']
df['emoticon_sum']= df['emoticon_1'] + df['emoticon_2'] + df['emoticon_3']
df['post_id_freq']= df.groupby('post_id')['post_id'].transform('count')`,
    points: [
      { label: "if_1 * if_2", desc: "Two columns multiply → combined signal stronger, non-linear pattern capture aagum" },
      { label: "if2_is_4", desc: "EDA-la: if_2==4 ila 98% Label 0 — very strong pattern, specific flag" },
      { label: "vote_diff", desc: "Upvote - Downvote. Negative = controversial comment" },
      { label: "vote_ratio", desc: "upvote / (downvote+1). +1 for zero division avoid" },
      { label: "emoticon_sum", desc: "3 emoticon columns sum. Emoticon presence = informal/emotional" },
      { label: "post_id_freq", desc: "Anda post-la evvalavu comments? High = popular/controversial topic" },
    ]
  },
  {
    num: 8,
    title: "TF-IDF Vectorizer",
    emoji: "📊",
    tag: "NLP",
    tagColor: "bg-green-50 text-green-600",
    code: `tfidf_word = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=3, max_df=0.95,
    sublinear_tf=True,
    analyzer='word',
    token_pattern=r'\\w{2,}',
)
tfidf_char = TfidfVectorizer(
    max_features=30000,
    ngram_range=(3, 5),
    min_df=5, max_df=0.95,
    sublinear_tf=True,
    analyzer='char_wb',
)
tfidf_word.fit(all_text)
tfidf_char.fit(all_text)`,
    points: [
      { label: "max_features=50000", desc: "Top 50k most useful words/phrases மட்டும் keep pannrom" },
      { label: "ngram_range=(1,2)", desc: "Unigram + bigram. 'bad movie' → ['bad','movie','bad movie']" },
      { label: "min_df=3", desc: "Minimum 3 docs-la irukkanum. Rare typos remove aagum" },
      { label: "max_df=0.95", desc: "95% docs-la iruntha — too common, useless. Skip pannrom" },
      { label: "sublinear_tf=True", desc: "log(tf+1) use pannrom. Frequent words scale down aagum" },
      { label: "analyzer='char_wb'", desc: "Char 3-5 grams. Misspelling, slang, tone capture aagum!" },
      { label: "fit(all_text)", desc: "Train+Test combined-la vocabulary learn pannrom. Both words include!" },
    ]
  },
  {
    num: 9,
    title: "Sparse Matrix Combine",
    emoji: "🔀",
    tag: "Data Prep",
    tagColor: "bg-blue-50 text-blue-600",
    code: `X_struct_sp_tr = sp.csr_matrix(X_struct_tr)
X_struct_sp_te = sp.csr_matrix(X_struct_te)

X_tr = sp.hstack([X_struct_sp_tr, X_word_tr, X_char_tr], format='csr')
X_te = sp.hstack([X_struct_sp_te, X_word_te, X_char_te], format='csr')

print(f"Final feature matrix: {X_tr.shape}")`,
    points: [
      { label: "sp.csr_matrix", desc: "Dense → Sparse format. Zeros store pannaa RAM waste — sparse-la skip pannrom" },
      { label: "sp.hstack", desc: "3 matrices horizontally join. Columns side by side paste pannrom" },
      { label: "Final shape", desc: "[36 structured | 50000 word | 30000 char] = ~80036 total features!" },
      { label: "Yen sparse?", desc: "Dense = 10GB RAM. Sparse = 100MB RAM. Memory huge save!" },
      { label: "format='csr'", desc: "Compressed Sparse Row — LightGBM-ku perfect format" },
    ]
  },
  {
    num: 10,
    title: "Class Weights",
    emoji: "⚖️",
    tag: "Imbalance Fix",
    tagColor: "bg-orange-50 text-orange-600",
    code: `# Labels: 0=57.7%, 1=8.0%, 2=31.5%, 3=2.8%
class_weight = {0: 1.0, 1: 4.0, 2: 1.5, 3: 8.0}
sample_weights = np.array([class_weight[label] for label in y])`,
    points: [
      { label: "Class imbalance", desc: "Label 3 = only 2.8%! Without weights model ignore pannudum" },
      { label: "Label 3 → 8x", desc: "'Inda sample 8 times important!' nu model-ku solrom. Rare class boost" },
      { label: "Label 0 → 1x", desc: "57.7% majority — already model learn pannudum, extra weight vendaam" },
      { label: "sample_weights", desc: "Every training row-ku oru weight assign pannrom" },
      { label: "Real world analogy", desc: "Exam-la tough questions 8 marks, easy questions 1 mark — same concept!" },
    ]
  },
  {
    num: 11,
    title: "LightGBM 5-Fold CV",
    emoji: "🔁",
    tag: "Modelling",
    tagColor: "bg-red-50 text-red-600",
    code: `skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds  = np.zeros((len(tr), 4))
test_preds = np.zeros((len(te), 4))

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_tr, y)):
    X_trn, X_val = X_tr[trn_idx], X_tr[val_idx]
    dtrain = lgb.Dataset(X_trn, label=y_trn, weight=w_trn)

    model = lgb.train(
        lgb_params, dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50)],
    )
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_te) / 5`,
    points: [
      { label: "StratifiedKFold(5)", desc: "Data-ai 5 parts. Each fold-la same label % maintain (class 3 missing aagaadu)" },
      { label: "random_state=42", desc: "Fixed seed = reproducible. Anyone run pannaalum same result" },
      { label: "oof_preds zeros((n,4))", desc: "4 class probabilities store. Initially all zeros" },
      { label: "early_stopping(50)", desc: "50 rounds improvement illana stop. Overfit avoid!" },
      { label: "num_boost_round=1000", desc: "Max 1000 decision trees build pannrom" },
      { label: "test_preds / 5", desc: "5 models average → ensemble → single model than better result!" },
    ]
  },
  {
    num: 12,
    title: "Generate Submission",
    emoji: "📤",
    tag: "Output",
    tagColor: "bg-teal-50 text-teal-600",
    code: `oof_labels = np.argmax(oof_preds, axis=1)
print(f"OOF Macro F1: {f1_score(y, oof_labels, average='macro'):.4f}")

for i, score in enumerate(f1_score(y, oof_labels, average=None)):
    print(f"  Label {i}: {score:.4f}")

final_labels = np.argmax(test_preds, axis=1)
sub['label'] = final_labels
sub.to_csv('submission.csv', index=False)`,
    points: [
      { label: "np.argmax(axis=1)", desc: "[0.05, 0.02, 0.90, 0.03] → 2. Row-wise highest probability class select" },
      { label: "f1_score average='macro'", desc: "Each class F1 → average. Imbalanced data-ku macro F1 best metric" },
      { label: "f1_score average=None", desc: "Per class F1. Class 3 score low iruntha identify pannrom" },
      { label: "argmax(test_preds)", desc: "Test rows probability → final label convert" },
      { label: "index=False", desc: "Row numbers save pannaa submission format wrong — skip pannrom" },
    ]
  },
];

const tagColors = {
  "Setup": "bg-slate-100 text-slate-600",
  "Data Prep": "bg-blue-50 text-blue-600",
  "NLP": "bg-emerald-50 text-emerald-700",
  "Feature Eng": "bg-violet-50 text-violet-600",
  "Imbalance Fix": "bg-orange-50 text-orange-600",
  "Modelling": "bg-rose-50 text-rose-600",
  "Output": "bg-teal-50 text-teal-600",
};

export default function App() {
  const [active, setActive] = useState(0);
  const s = steps[active];

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-3xl mx-auto">

        {/* Header */}
        <div className="mb-6">
          <h1 className="text-2xl font-semibold text-gray-800 mb-1">Comment Category Prediction</h1>
          <p className="text-sm text-gray-400">Step-by-step code walkthrough — {steps.length} steps total</p>
        </div>

        {/* Step Pills */}
        <div className="flex flex-wrap gap-2 mb-5">
          {steps.map((st, i) => (
            <button
              key={i}
              onClick={() => setActive(i)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border transition-all ${
                active === i
                  ? "bg-gray-800 text-white border-gray-800 shadow-sm"
                  : "bg-white text-gray-500 border-gray-200 hover:border-gray-400 hover:text-gray-700"
              }`}
            >
              <span>{st.emoji}</span>
              <span>{st.num}</span>
            </button>
          ))}
        </div>

        {/* Main Card */}
        <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">

          {/* Card Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100">
            <div className="flex items-center gap-3">
              <span className="text-2xl">{s.emoji}</span>
              <div>
                <h2 className="text-base font-semibold text-gray-800">{s.title}</h2>
                <p className="text-xs text-gray-400">Step {s.num} of {steps.length}</p>
              </div>
            </div>
            <span className={`text-xs font-medium px-3 py-1 rounded-full ${s.tagColor}`}>
              {s.tag}
            </span>
          </div>

          {/* Code Block */}
          <div className="relative">
            <div className="flex items-center gap-1.5 px-4 py-2.5 bg-gray-900 border-b border-gray-700">
              <div className="w-3 h-3 rounded-full bg-red-400"></div>
              <div className="w-3 h-3 rounded-full bg-yellow-400"></div>
              <div className="w-3 h-3 rounded-full bg-green-400"></div>
              <span className="ml-2 text-xs text-gray-400 font-mono">solution.py</span>
            </div>
            <pre className="bg-gray-900 text-gray-100 text-xs leading-relaxed p-5 overflow-x-auto font-mono m-0">
              {s.code}
            </pre>
          </div>

          {/* Explanation Section */}
          <div className="px-6 py-5">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-1 h-4 bg-gray-800 rounded-full"></div>
              <h3 className="text-sm font-semibold text-gray-700">Line-by-line explanation</h3>
            </div>
            <div className="space-y-2">
              {s.points.map((p, i) => (
                <div key={i} className="flex gap-3 p-3 rounded-xl bg-gray-50 border border-gray-100 hover:border-gray-300 transition-colors">
                  <code className="text-xs font-mono text-indigo-600 whitespace-nowrap pt-0.5 min-w-0 shrink-0" style={{maxWidth: 220, overflow: "hidden", textOverflow: "ellipsis"}}>
                    {p.label}
                  </code>
                  <span className="text-gray-500 text-xs pt-0.5">—</span>
                  <p className="text-sm text-gray-700 leading-relaxed m-0">{p.desc}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Progress + Nav */}
          <div className="px-6 py-4 border-t border-gray-100 flex items-center justify-between">
            <button
              onClick={() => setActive(a => Math.max(0, a - 1))}
              disabled={active === 0}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium border transition-all ${
                active === 0
                  ? "opacity-30 cursor-not-allowed border-gray-200 text-gray-400"
                  : "border-gray-200 text-gray-600 hover:bg-gray-50 hover:border-gray-300"
              }`}
            >
              ← Previous
            </button>

            {/* Progress Bar */}
            <div className="flex-1 mx-5">
              <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gray-800 rounded-full transition-all duration-300"
                  style={{ width: `${((active + 1) / steps.length) * 100}%` }}
                />
              </div>
              <p className="text-center text-xs text-gray-400 mt-1">{active + 1} / {steps.length}</p>
            </div>

            <button
              onClick={() => setActive(a => Math.min(steps.length - 1, a + 1))}
              disabled={active === steps.length - 1}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium border transition-all ${
                active === steps.length - 1
                  ? "opacity-30 cursor-not-allowed border-gray-200 text-gray-400"
                  : "bg-gray-800 text-white border-gray-800 hover:bg-gray-700"
              }`}
            >
              Next →
            </button>
          </div>
        </div>

        {/* Step overview footer */}
        <div className="mt-4 grid grid-cols-6 gap-2">
          {steps.map((st, i) => (
            <button
              key={i}
              onClick={() => setActive(i)}
              className={`p-2 rounded-xl border text-center transition-all ${
                active === i
                  ? "border-gray-800 bg-gray-800 text-white"
                  : i < active
                  ? "border-gray-300 bg-gray-100 text-gray-500"
                  : "border-gray-200 bg-white text-gray-400 hover:border-gray-300"
              }`}
            >
              <div className="text-base">{st.emoji}</div>
              <div className="text-xs mt-0.5 font-medium">{st.num}</div>
            </button>
          ))}
        </div>

      </div>
    </div>
  );
}