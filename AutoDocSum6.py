path='Claim_Req_Travelers_McDonald_5_14.pdf'
import subprocess
import matplotlib
import mglearn
import numpy
import pandas
import pdfminer
import sklearn
import sys
from io import StringIO
#Python LDA Visualization Libraries
import pyLDAvis
import pyLDAvis.sklearn
import PyPDF2
# pdfminer
from pdfminer.converter import PDFPageAggregator
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.layout import LTTextBoxHorizontal
from pdfminer.pdfdevice import PDFDevice
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdftypes import resolve1
#Relevant Analysis Librariess
import re
import numpy as np
import pandas as pd
import mglearn as mg
#Relevant Modeling Libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
# %pylab
# %matplotlib inline

#Creating a Function to Read the PDF Document and convert into Text
def convert_pdf_to_text(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos = set()
    
    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching, check_extractable=True):
        interpreter.process_page(page)
        
    text = retstr.getvalue()
    
    fp.close()
    device.close()
    retstr.close()
    
    return text
#Reading the PDF Document and saving as lone
lone=convert_pdf_to_text('C:/Users/matth/Documents/Jobs/Travelers/NLP/Automated_Doc_Summarization_NLP/Claim_Req_Travelers_McDonald_5_14.pdf')

# Gets inputs rb
fp = open(path, 'rb')
parser = PDFParser(fp)
doc = PDFDocument(parser)
fields = resolve1(doc.catalog['AcroForm'])['Fields']

inps=[]
for i in fields:
    field = resolve1(i)
    name, value = field.get('T'), field.get('V')
    inps.append('{0}: {1}'.format(name, value))
    
inf=[]
ini=[]
for i in fields:
    field=resolve1(i)
    name,value=field.get('T'),field.get('V')
    inf.append(name)
    ini.append(value)

# Topic Modeling
# Fitting Count Vectorizer on the document with Stop Words
vect=CountVectorizer(ngram_range=(1,1),stop_words='english')
dtm = vect.fit_transform(inps)

#Converting the Document Term Matrix from Count Vectorizer into a Pandas Dataframe
dfm=pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names())

#Fitting the Latent Dirichlet Allocation Model on the Document Term Matrix
lda = LatentDirichletAllocation(n_components=5)
lda_dtf = lda.fit_transform(dtm)
#Latent Dirichlet Allocation Model
# lda_dtf

# Topic Extracting
#Extracting 5 Topics from LDA and the most common words in each topic
sorting = np.argsort(lda.components_)[:, ::-1]
features = np.array(vect.get_feature_names())

mg.tools.print_topics(topics=range(5), feature_names=features, sorting=sorting, topics_per_chunk=5, n_words=15)

#Sentences within the Topic Model 1
topic_0 = np.argsort(lda_dtf[:,0])[::-1]
# for i in topic_0[:5]:
#     print(f".".join(inps[i].split(f".")[:2]) + f".\n")
    
#Senteces within the Topic Model 2
topic_1 = np.argsort(lda_dtf[:,1])[::-1]
# for i in topic_1[:5]:
#     print(f".".join(inps[i].split(f".")[:2]) + f".\n")

#Senteces within the Topic Model 3
topic_2 = np.argsort(lda_dtf[:,2])[::-1]
# for i in topic_2[:5]:
#     print(f".".join(inps[i].split(f".")[:2]) + f".\n")
    
#Senteces within the Topic Model 4
topic_3 = np.argsort(lda_dtf[:,3])[::-1]
# for i in topic_3[:5]:
#     print(f".".join(inps[i].split(f".")[:2]) + f".\n")
    
#Senteces within the Topic Model 5
topic_4 = np.argsort(lda_dtf[:,4])[::-1]
# for i in topic_4[:5]:
#     print(f".".join(inps[i].split(f".")[:2]) + f".\n")
    
# Topic Visualization
# zit=pyLDAvis.sklearn.prepare(lda,dtm,vect)
# pyLDAvis.display(zit)





