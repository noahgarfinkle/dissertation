from lxml import etree as ET

# http://lxml.de/tutorial.html
xmlPath = "./input.xml"

class SiteSearch:
    def __init__(self):
        return None

class SiteRelationalConstraint:
    def __init__(self):
        return None

class Input:
    def __init__(self,xmlPath):
        self.xmlPath = xmlPath
        self.tree = ET.parse(xmlPath)
        self.root = tree.getroot()
        self.siteSearches = []
        self.siteRelationalConstraints = []
        self.resultDir = root.attrib['resultDir']
        self.studyObjectiveID = root.attrib['studyObjectiveID']

    def pretty_print(self):
        print ET.tostring(self.root,pretty_print=True)

    def retrieveSiteSearches(self):
        siteSearches = self.root[0]
        for siteSearchElement in siteSearches:
            print "%s: %s" %(siteSearchElement.tag, siteSearchElement.attrib)
            for siteSearchElementChild in siteSearchElement:
                print "\t%s: %s" %(siteSearchElementChild.tag,siteSearchElementChild.attrib)
            siteSearch = SiteSearch()
            self.siteSearches.append(siteSearch)

    def retrieveSiteRelationalConstraints(self):
        siteRelationalConstraints = self.root[1]
        siteRelationalConstraint = SiteRelationalConstraint()
        self.siteRelationalConstraints.append(siteRelationalConstraint)
