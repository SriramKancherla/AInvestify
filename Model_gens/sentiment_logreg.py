import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import csv
import joblib


positive_words = {
    # General positive
    "good","great","excellent","positive","optimistic","favorable","strong","solid",
    "improve","improving","improved","recovery","recover","rebound","momentum",

    # Financial performance
    "profit","profits","profitable","profitability","earn","earnings","revenue",
    "growth","expansion","margin","margins","cashflow","freecashflow",
    "record","recordhigh","alltimehigh","highs","surge","jump","rally",
    "beat","outperform","outperformance","upside","upgrade",

    # Stock market terms
    "bullish","bull","breakout","uptrend","higher","rise","rising","gains",
    "gain","up","green","accumulate","accumulation","support",

    # Analyst / institutional
    "buy","strongbuy","overweight","topick","conviction",
    "institutionalbuying","fundinflows",

    # Corporate actions
    "dividend","dividends","dividendgrowth","buyback","sharebuyback",
    "merger","acquisition","strategic","partnership","deal","contract",

    # Business strength
    "leadership","dominant","moat","competitive","innovation","innovative",
    "scalable","efficiency","efficient","costcutting","costreduction",

    # Macro / guidance
    "raisedguidance","guidanceup","forecastup","tailwinds",
    "demand","strongdemand","pricingpower",

    # Tech / growth buzzwords
    "ai","artificialintelligence","cloud","datacenter","semiconductor",
    "automation","digital","platform","subscription","recurring",

    # Risk sentiment
    "confidence","confident","stability","stable","resilient","resilience"
}

negative_words = {
    # General negative
    "bad","poor","negative","weak","weaker","weakness","decline","deterioration",
    "uncertain","uncertainty","pessimistic","risk","risky","volatile","volatility",

    # Financial trouble
    "loss","losses","unprofitable","miss","missed","shortfall","deficit",
    "burn","cashburn","debt","leverage","liquidity","default",

    # Stock market terms
    "bearish","bear","sell","selloff","selling","dump","crash","collapse",
    "drop","fall","fallen","plunge","plunged","down","red","lower","slide",
    "downtrend","breakdown","resistance",

    # Analyst / institutional
    "downgrade","underperform","underweight","cutrating","sellrating",
    "pricecut","targetcut","exit",

    # Corporate / legal
    "lawsuit","fraud","scandal","probe","investigation","regulatory",
    "fine","penalty","ban","shutdown","recall",

    # Business weakness
    "slowdown","slow","shrinking","contraction","layoff","layoffs",
    "jobcuts","restructuring","bankruptcy","insolvency",

    # Guidance / outlook
    "loweredguidance","guidancedown","warning","profitwarning",
    "headwinds","pressure","marginpressure",

    # Macro / economy
    "recession","inflation","stagflation","ratehike","tightening",
    "tariff","tradeban","geopolitical","war","conflict",

    # Tech / growth risks
    "competition","disruption","obsolete","decliningusers",
    "churn","securitybreach","hack","outage",

    # Sentiment
    "fear","panic","concern","worry","worried","sellpressure"
}



def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"&amp;", "and", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def auto_label(text):
    score=0
    for word in text.split():
        if word in positive_words:
            score+=1
        if word in negative_words:
            score-=1
    if score>0:
        return 1
    elif score<0:
        return 0
    else:
        return None





vectorizer=TfidfVectorizer(max_features=5000,stop_words='english',ngram_range=(1,3))


data=pd.read_csv('../datasets/stock_tweets.csv',sep='\t',        
    engine='python',
    quoting=csv.QUOTE_NONE,   
    on_bad_lines='skip',)

data['Tweet']=data['Tweet'].apply(clean_tweet)
data['label']=data['Tweet'].apply(auto_label)
data = data.dropna(subset=['label'])


X=data['Tweet']
Y=data['label']
X=vectorizer.fit_transform(X)

model=LogisticRegression(max_iter=1000)
model.fit(X,Y)
joblib.dump(model,"sentiment_logreg.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
