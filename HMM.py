from nltk import defaultdict
import numpy as np
from nltk.corpus import brown


class HMM():
    def __init__(self,corpus=None,alpha=0.1):
        # if corpus!=None:
        self.c_uvs=defaultdict(lambda :0)
        self.c_uv=defaultdict(lambda :0)
        self.c_xs=defaultdict(lambda :0)
        self.c_s=defaultdict(lambda :0)
        self.alpha=alpha

    def fit(self,corpus,format_flag=True,alpha=0.1):
        if alpha!=0:
            self.alpha=alpha

        def preproccess_corpus(corpus):
            # This function will take each sentence then "process each sentence"

            def proccess_sentence(sent):
                new_sent = []
                new_sent.append(("START", "*"))
                new_sent.append(("START", "*"))
                for w, l in sent:
                    new_sent.append((w, l))
                new_sent.append(("STOP", "END"))
                return new_sent

            new_corpus = []
            for sent in corpus:
                new_sent = proccess_sentence(sent)
                new_corpus.append(new_sent)
            return new_corpus

        if format_flag==True:
            corpus=preproccess_corpus(corpus)

        def distribution(corpus):
            c_uvs=defaultdict(lambda :0)
            c_uv=defaultdict(lambda :0)
            c_xs=defaultdict(lambda :0)
            c_s=defaultdict(lambda :0)
            K_l=set()
            for sent in corpus:
                for iter, val in enumerate(sent):
                    # Loop for count(uv)
                    if iter >= 1:
                        u = sent[iter - 1][1]
                        v = sent[iter][1]
                        c_uv[u + "_" + v] += 1

                for iter, val in enumerate(sent):
                    # Loop for count(suv)
                    if iter >= 2:
                        u = sent[iter - 2][1]
                        v = sent[iter - 1][1]
                        s = sent[iter][1]
                        c_uvs[u + "_" + v + "_" + s] += 1

                for iter,val in enumerate(sent):
                    #Loop for count xs
                    if iter>=2:
                        x=val[0]
                        s=val[1]
                        c_xs[x+"_"+s]+=1

                for iter,val in enumerate(sent):
                    #Loop for count s
                    if iter>=2:
                        s=val[1]
                        c_s[s]+=1

                for iter,val in enumerate(sent):
                    K_l.add(val[1])
            return c_uvs,c_uv, c_xs,c_s,list(K_l)
        self.c_uvs,self.c_uv,self.c_xs,self.c_s,self.K_l=distribution(corpus)

        def smoothing_q(K_l,c_uvs,alpha):
            for u in K_l:
                for v in K_l:
                    for s in K_l:
                        uvs=u+"_"+v+"_"+s
                        c_uvs[uvs]+=alpha
            return c_uvs
        self.c_uvs =smoothing_q(self.K_l,self.c_uvs,self.alpha)


    def predict(self,sent,format_flag=True):

        if len(sent)<3:
            prediction=[]
            for i in range(len(sent)):
                prediction.append("NOUN")

            return prediction
        if format_flag==True:
            org_sent=sent.copy()

            tmp=[]
            tmp.append("START")
            tmp.append("START")
            tmp.extend(sent)
            tmp.append("STOP")

            sent=tmp
        else:
            org_sent=sent[2:-1]
        def smoothing_e(sent,K_l,c_xs,alpha):
            for ix in range(2,len(sent)-1):
                for s in K_l:
                    x=sent[ix]
                    xs=x+"_"+s
                    c_xs[xs]+=alpha
            return c_xs
        self.c_xs=smoothing_e(sent,self.K_l,self.c_xs,self.alpha)

        def probability(c_uvs,c_uv,c_xs,c_s,alpha):
            q_prob=defaultdict()
            e_prob=defaultdict()
            for uvs in c_uvs.keys():
                tmp=uvs.split("_")
                u=tmp[0]
                v=tmp[1]
                uv=u+"_"+v
                q_prob[uvs]=float(c_uvs[uvs])/(c_uv[uv]+alpha*len(c_uvs))

            for xs in c_xs.keys():
                tmp=xs.split("_")
                s=tmp[1]
                e_prob[xs]=float(c_xs[xs])/(c_s[s]+alpha*len(c_xs))

            return q_prob,e_prob
        self.q_prob,self.e_prob=probability(self.c_uvs,self.c_uv,self.c_xs,
                                            self.c_s,self.alpha)

        self.pi=[]
        self.bp=[]
        for i in range(len(org_sent)+1):
            self.pi.append(dict())
            self.bp.append(dict())
            for j in self.K_l:
                self.pi[i][j]=dict()
                self.bp[i][j]=dict()
                for t in self.K_l:
                    self.pi[i][j][t]=0
                    self.bp[i][j][t]=""
        self.pi[0]["*"]["*"]=1

        ######Viterbri
        for i in range(1,len(org_sent)+1):
            for u in self.K_l:
                for v in self.K_l:
                    result_list=[]
                    result_label=[]
                    for w in self.K_l:
                        wuv=w+"_"+u+"_"+v
                        x=org_sent[i-1]
                        xv=x+"_"+v
                        score=self.pi[i-1][w][u]*self.q_prob[wuv]*self.e_prob[xv]
                        result_list.append(score)
                        result_label.append(w)
                    self.pi[i][u][v]=np.max(result_list)
                    self.bp[i][u][v]=result_label[np.argmax(result_list)]

        prediction=[]
        for i in range(len(org_sent)):
            prediction.append("")

        result_tmp=[]
        result_label_tmp=[]
        for u in self.K_l:
            for v in self.K_l:
                result_tmp.append(self.pi[len(org_sent)][u][v]*self.q_prob[u+"_"+v+"_"+"END"])
                result_label_tmp.append((u,v))
        a1,a2=result_label_tmp[np.argmax(result_tmp)]

        prediction[-1]=a2
        prediction[-2]=a1

        for i in reversed(range(len(prediction) - 2)):
            prediction[i] = self.bp[i + 3][prediction[i + 1]][prediction[i + 2]]

        return prediction

    def predict_many(self,sents,format_flag=True):
        predictions=[]
        for sent in sents:
            predictions.append(self.predict(sent,format_flag=format_flag))
        return predictions
