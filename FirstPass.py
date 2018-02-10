# -*- coding: utf-8 -*-
"""Parses an XML file in order to evaluate sites
"""

__author__ = "Noah W. Garfinkle"
__copyright__ = "Copyright 2018, Noah W. Garfinkle"
__credits__ = ["Dr. Ximing Cai", "Dr. George Calfas", "Thomas 'Max' Foltz",
                    "Juliana McMillan-Wilhoit", "Matthew Hiett",
                    "Dylan Pasley"]
__license__ = "GPL"
__version__ = "0.0.1"
__version_dinosaur__ = "Apotosauras"
__maintainer__ = "Noah Garfinkle"
__email__ = "garfink2@illinois.edu"
__status__ = "Development"
__python_version__ = "2.7"
__date_created__ = "10 FEBRUARY 2018"

# IMPORTS
from lxml import etree as ET

""" REFERENCES
http://lxml.de/tutorial.html
"""


# CLASSES
class SiteSearch:
    def __init__(self):
        self.crs = None
        self.AreaOfInterest = None

class SiteSuitabilityCriteria:
    def __init__(self):
        return None

class SiteSuitabilityCriteriaEvaluation:
    def __init__(self):EvalFallBack
        return None

class SiteRelationalConstraint:
    def __init__(self):
        return None

class Input:
    def __init__(self,xmlPath):
        self.xmlPath = xmlPath
        self.tree = ET.parse(xmlPath)
        self.root = tree.getroot()
        self.siteSearches = self.retrieveSiteSearches()
        self.siteRelationalConstraints = self.retrieveSiteRelationalConstraints()
        self.resultDir = root.attrib['resultDir']
        self.studyObjectiveID = root.attrib['studyObjectiveID']

    def pretty_print(self):
        print ET.tostring(self.root,pretty_print=True)

    def retrieveSiteSearches(self):
        siteSearches = []
        siteSearchesElement = self.root[0]
        for siteSearchElement in siteSearchesElement:
            print "%s: %s" %(siteSearchElement.tag, siteSearchElement.attrib)
            for siteSearchElementChild in siteSearchElement:
                print "\t%s: %s" %(siteSearchElementChild.tag,siteSearchElementChild.attrib)
            siteSearch = SiteSearch()
            siteSearches.append(siteSearch)
        return siteSearches

    def retrieveSiteRelationalConstraints(self):
        siteRelationalConstraints = self.root[1]
        siteRelationalConstraint = SiteRelationalConstraint()
        siteSearchRelationalConstraints = []
        siteRelationalConstraints.append(siteRelationalConstraint)
        return siteSearchRelationalConstraints

# TESTS
xmlPath = "./input.xml"
