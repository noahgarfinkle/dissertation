# -*- coding: utf-8 -*-
"""
Parses an XML file and creatse a Jupyter notebook
"""

__author__ = "Noah W. Garfinkle"
__copyright__ = "Copyright 2018, Noah W. Garfinkle"
__credits__ = ["Dr. Ximing Cai", "Dr. George Calfas", "Thomas 'Max' Foltz",
                    "Juliana McMillan-Wilhoit", "Matthew Hiett",
                    "Dylan Pasley", "Marcus Voegle", "Eric Kreiger"]
__license__ = "GPL"
__version__ = "0.0.1"
__version_dinosaur__ = "Apotosauras"
__maintainer__ = "Noah Garfinkle"
__email__ = "garfink2@illinois.edu"
__status__ = "Development"
__python_version__ = "2.7"
__date_created__ = "24 FEBRUARY 2018"

## IMPORTS
import nbformat as nbf

""" REFERENCES
https://gist.github.com/fperez/9716279
"""

class Notebook:
    def __init__(self):
        self.notebook = nbf.v4.new_notebook()

    def addMarkdown(self,text,headerLevel=0):
        if headerLevel > 0:
            prepend = ""
            for i in range(0,headerLevel + 1):
                prepend = prepend + "#"
            text = prepend + " " + text
        self.notebook['cells'].append(nbf.v4.new_markdown_cell(text))

    def saveNotebook(self,filePath):
        with open(filePath, 'w') as f:
            nbf.write(self.notebook, f)

    def runNotebook(self):
        return None

notebook = Notebook()
notebook.addMarkdown("Test notebook from a class")
notebook.saveNotebook("./results/acreatednotebook.ipynb")

nb = nbf.v4.new_notebook()
text = """\
# My first automatic Jupyter Notebook
This is an auto-generated notebook."""

code = """\
%pylab inline
hist(normal(size=2000), bins=50);"""

nb['cells'] = [nbf.v4.new_markdown_cell(text),
               nbf.v4.new_code_cell(code) ]

nb['cells'].append(nbf.v4.new_code_cell(code))


fname = 'test.ipynb'

with open(fname, 'w') as f:
    nbf.write(nb, f)

nb['cells']
