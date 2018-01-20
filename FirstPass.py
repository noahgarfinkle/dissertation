dataDir = ""
resultDir = ""
lat_min = -4.12933991491484
lon_min = 39.3832397460938
lat_max = -3.86151427357038
lon_max = 39.7416687011719
qafCellSize = 857
populationDensityWeight = 0.2
slopeWeight = 0.2
msrWeight = 0.2
apodWeight = 0.2

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
