import spacy
import seaborn as sb
import matplotlib.pyplot as plt
from processing import tf_idf_scores
from data import t0, t1, t2, t3, t4, t5, t6
nlp = spacy.load("en")

doc0 = nlp(t0)
doc1 = nlp(t1)
doc2 = nlp(t2)
doc3 = nlp(t3)
doc4 = nlp(t4)
doc5 = nlp(t5)
doc6 = nlp(t6)
doc_list = [doc0, doc1, doc2, doc3, doc4, doc5, doc6]


df1 = tf_idf_scores(doc_list)
df2 = tf_idf_scores(doc_list, scale=True)


dims = (20, 16)
fig, ax = plt.subplots(nrows=2, figsize=dims)
sb.heatmap(df1, linewidths=.004, ax=ax[0]).set_title(
 "TF-IDF Heatmap", fontsize=18)
sb.heatmap(df2, linewidths=.004, ax=ax[1]).set_title(
 "TF-IDF Heatmap - scaled", fontsize=18)
plt.show()
