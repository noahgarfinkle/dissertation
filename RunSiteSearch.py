import sys

import SpatialIO as io
import FirstPass as firstpass
#import pgdissroute as pgdissroute

def main():
    xmlPath = str(sys.argv[1:][0])
    layerIDs = firstpass.runMSSPIX(xmlPath)
    singleLayerID = layerIDs[0]
    return singleLayerID

if __name__ == "__main__":
    layerid = main()
    print layerid
