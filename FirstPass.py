from lxml import etree as ET

class InputFile:
    def __init__(self,xmlPath):
        self.xmlPath = xmlPath



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




print ET.tostring(root,pretty_print=True)

def readXML(xmlPath):
    tree = ET.parse(xmlPath)
    root = tree.getroot()

def firstPass(dataDir,resultDir,lat_min,lon_min,lat_max,lon_max,qafCellSize,populationDensityWeight,slopeWeight,msrWeight,apodWeight):
    return 0

def createQAFRaster(qafCellSize,lx,ly,ux,uy):
    return 0

def loadLayerFromSource(filePath):
    return 0

def containsPolygon(polygonList,qafRaster):
    return 0

def customResample(inputRaster,qafRaster,fun="max"):
    return 0

def filterFeatureLayer(featureLayer,column,value):
    return 0

def pointDistance(qafRaster,pointsLayer):
    return 0

def lineDistance(qafRaster,lineLayer):
    return 0

def createExclusionRaster(exclusionFeaturePaths,qafRaster):
    return 0

def loadFeatureLayerFromShapeFile(featurePath):
    reutrn 0

def weightRaster(raster,weight):
    return 0

def reclassifyRaster(rasterToReclassify,reclassificationCSVPath):
    return 0

def reclassifyRasterFromString(rasterToReclassify,reclassificationString):
    return 0

def uniqueValueReclassify(rasterToSubstitute,list_valuesToReplace,list_valuesToReplaceWith):
    return 0

def cropRasterToAOI(rasterToCrop,qafRaster):
    return 0
