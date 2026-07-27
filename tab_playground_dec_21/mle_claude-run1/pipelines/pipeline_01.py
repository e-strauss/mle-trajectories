"""pipeline_01 -- trivial floor baseline: predict the majority class always.

Establishes the accuracy floor (majority class 2 is 56.55% of rows), so every
later model is judged against "did we beat always-guess-2".
"""
from sklearn.dummy import DummyClassifier

from common import load_xy

X, y = load_xy(target="Cover_Type")
X = X.skb.drop(cols="Id")
pred = X.skb.apply(DummyClassifier(strategy="most_frequent"), y=y)

DESCRIPTION = "Majority-class DummyClassifier (accuracy floor)"
