from nltk.corpus import brown
from HMM import HMM


tagged_corpus=brown.tagged_sents(categories=["adventure"],tagset="universal")

model=HMM()
model.fit(tagged_corpus)

x=["The","man","certainly","didn't","want","to","wait"]
pred=model.predict(x)
print(pred)
####### -> Seem work very well :O :O :O


####### Evaluate the performance of the TriHMM

eval_sents=brown.tagged_sents(categories=["romance"],tagset="universal")

x=[[ w for w,t in sent] for sent in eval_sents ]
y_actual=[[t for w,t in sent] for sent in eval_sents]

y_predict=model.predict_many(x)

correct=0
total=0
for iter,sent in enumerate(y_actual):
    for jter,val in enumerate(sent):
        if y_actual[iter][jter]==y_predict[iter][jter]
            correct+=1
        total+=1
print(float(correct)/total)

#######Evaluation will spend little long time

