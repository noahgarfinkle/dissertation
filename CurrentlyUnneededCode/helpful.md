# Moving between Jupyter, Mardown, and Word
* https://blog.ouseful.info/2017/06/13/using-jupyter-notebooks-for-assessment-export-as-word-docx-extension/
* https://pandoc.org/demos.html
* pandoc -s example30.docx -t markdown -o example35.md

# Tracking time for a function
```Python
start = datetime.datetime.now()
end = datetime.datetime.now()
end - start
timeElapsed = end - start
initialFeatures = len(dfToFilter.index)
filteredFeatures = len(filteredDF.index)
print "%s %s of %s candidates in %s seconds" %(returnText,filteredFeatures,initialFeatures,timeElapsed.seconds)
```
