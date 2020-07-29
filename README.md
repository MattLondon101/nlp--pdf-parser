# Natural_Language_Processing_Document_Parser
In AutoDocSum6.py, replace path (1st line) with path to your PDF. Then run AutoDocSum6.py to see keywords from you PDF grouped into 5 distinct groups.

For interactive chart, inset AutoDocSum6.py into a JupyterNotebook cell, adding:
```
zit=pyLDAvis.sklearn.prepare(lda,dtm,vect)  
pyLDAvis.display(zit)
```
to a separate cell below.  
Run both cells to view interactive chart for keyword groups.

View the Jupyter Notebook @ [nbviewer](https://nbviewer.jupyter.org/github/MattLondon101/Images/blob/master/AutoDocSum6.ipynb)

**Download Claim_Req_Travelers_McDonald_5_14.pdf to view the sample PDF that is summarized by AutoDocSum6.py**


![Topic Visualization at end of Notebook](https://github.com/MattLondon101/Images/blob/master/TopicVisualization1.png)
