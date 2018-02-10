from lxml import etree as ET

class SiteSearch:
    def __init__(self):
        return None

class siteRelationalConstraint:
    def __init__(self):
        return None

class Input:
    def __init__(self,xmlPath):
        self.xmlPath = xmlPath
        self.tree = ET.parse(xmlPath)
        self.root = tree.getroot()

    def pretty_print(self):
        print ET.tostring(self.root,pretty_print=True)




# http://lxml.de/tutorial.html
xmlPath = "./input.xml"
tree = ET.parse(xmlPath)
root = tree.getroot()
root.tag
root.attrib

root.getchildren()
sites = root[0]

for child in sites:
    print "%s: %s" %(child.tag, child.attrib)
    for grandchild in child:
        print "\t%s: %s" %(grandchild.tag,grandchild.attrib)
