# -*- coding: utf-8 -*-
"""
Implements the spatial optimization algorithms
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
__date_created__ = "12 May 2018"

## IMPORTS
from lxml import etree as ET
import gdaltools as gdt
from osgeo import gdal, osr, ogr
import pandas as pd
import geopandas as gpd
import fiona
import pyproj
import rasterio
import numpy as np
import numpy.ma as ma
import math
import shapely
from shapely.geometry import Point, Polygon
from shapely.wkt import loads
import shapely.geometry as geom
import shapely.affinity
from rasterstats import zonal_stats, raster_stats, point_query, utils
import matplotlib.pyplot as plt
import datetime
import time
import random

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import CandidateDataFrameOperations as candidates
import ENSITEIO as eio
import Objective_Analytic as objective_analytic
import Objective_Raster as objective_raster
import pgdissroute as pgdissroute
import SpatialIO as io

## GLOBALS
toolbox = base.Toolbox()

## OBJECTIVE FUNCTIONS BETWEEN CANDIDATES
def evaluateCandidates_EuclideanDistance(df1,index1,df2,index2):
    try:
        geom1 = df1[index1:index1+1]
        geom1.crs = {'init':'EPSG:3857'}

        geom2 = df2[index2:index2+1]
        geom2.crs = {'init':'EPSG:3857'}

        euclideanDistance = geom2.distance(geom1.geometry[index1]).min()
        return euclideanDistance
    except Exception as e:
        print e
        print "Euclidean distance failed"

def evaluateCandidates_DrivingDistance(df1,index1,df2,index2):
    try:
        geom1 = df1[index1:index1+1]
        geom1.crs = {'init':'EPSG:3857'}
        geom1 = geom1.to_crs({'init':'EPSG:4326'})
        geom1 = geom1['geometry']

        geom2 = df2[index2:index2+1]
        geom2.crs = {'init':'EPSG:3857'}
        geom2 = geom2.to_crs({'init':'EPSG:4326'})
        geom2 = geom2['geometry']

        # reproject the geometries
        totalDriveDistance = pgdissroute.calculateRouteDistance(geom1.centroid.x[index1],geom1.centroid.y[index1],geom2.centroid.x[index2],geom2.centroid.y[index2])
        return totalDriveDistance
    except Exception as e:
        print e
        print "Driving distance failed"

## OBJECTIVE FUNCTIONS BETWEEN A CANDIDATE AND SOURCE

## SPATIAL OPTIMIZATION IMPLEMENTATION
def evaluate(individual,listOfDataFrames,siteRelationalConstraints):
    """
    individual is set of indices in the same order as the listOfDataFrames
    TODO: add caching
    TODO: add individual MCDA score as objective function
    """
    try:
        results = {}
        results['MCDA_SCORE'] = 0.0
        totalWeight = 0
        #results[individual] = ",".join([str(x) for x in individual])
        geneIndex = 0
        for gene in individual:
            colName = "gene_%s" %(geneIndex)
            results[colName] = gene
            geneIndex += 1
        siteRelationalConstraint_constraintNames = []
        for siteRelationalConstraint in siteRelationalConstraints:
            siteRelationalConstraint_constraintName = siteRelationalConstraint.attrib['constraintName']
            if siteRelationalConstraint.tag == "SiteRelationalConstraint_Routing":
                print "Site Relational Constraint: Routing"
                siteRelationalConstraint_site1Index = int(siteRelationalConstraint.attrib['site1Index'])
                siteRelationalConstraint_site2Index = int(siteRelationalConstraint.attrib['site2Index'])
                siteRelationalConstraint_site1DF = listOfDataFrames[siteRelationalConstraint_site1Index]
                siteRelationalConstraint_site2DF = listOfDataFrames[siteRelationalConstraint_site2Index]
                siteRelationalConstraint_site1Candidate = individual[siteRelationalConstraint_site1Index]
                siteRelationalConstraint_site2Candidate = individual[siteRelationalConstraint_site2Index]
                routingDistance = evaluateCandidates_DrivingDistance(siteRelationalConstraint_site1DF,siteRelationalConstraint_site1Candidate,siteRelationalConstraint_site2DF,siteRelationalConstraint_site2Candidate)
                results[siteRelationalConstraint_constraintName] = routingDistance
                scores = siteRelationalConstraint.find("Scores")
                weight = scores.attrib['weight']
                totalWeight += float(weight)
                isZeroExclusionary = scores.attrib['isZeroExclusionary']
                default = scores.attrib['default']
                scoreStructure = []
                for scoreRow in scores:
                    lowerBoundInclusive = str(scoreRow.attrib['lowerBoundInclusive'])
                    upperBoundExclusive = str(scoreRow.attrib['upperBoundExclusive'])
                    score = str(scoreRow.attrib['score'])
                    scoreSet = [lowerBoundInclusive,upperBoundExclusive,score]
                    scoreStructure.append(scoreSet)
                evaluationDF =  pd.DataFrame(results,index=[0])
                evaluationDF = candidates.scoreDF(evaluationDF,siteRelationalConstraint_constraintName,scoreStructure,isZeroExclusionary=False)
                qafName = "%s_QAF" %(siteRelationalConstraint_constraintName)
                siteRelationalConstraint_constraintNames.append(qafName)
                results[qafName] = evaluationDF[qafName][0]
                results['MCDA_SCORE'] += float(weight) * float(evaluationDF[qafName][0])
            if siteRelationalConstraint.tag == "SiteRelationalConstraint_Euclidean":
                print "Site Relational Constraint: Euclidean"
                siteRelationalConstraint_site1Index = int(siteRelationalConstraint.attrib['site1Index'])
                siteRelationalConstraint_site2Index = int(siteRelationalConstraint.attrib['site2Index'])
                siteRelationalConstraint_site1DF = listOfDataFrames[siteRelationalConstraint_site1Index]
                siteRelationalConstraint_site2DF = listOfDataFrames[siteRelationalConstraint_site2Index]
                siteRelationalConstraint_site1Candidate = individual[siteRelationalConstraint_site1Index]
                siteRelationalConstraint_site2Candidate = individual[siteRelationalConstraint_site2Index]
                euclideanDistance = evaluateCandidates_EuclideanDistance(siteRelationalConstraint_site1DF,siteRelationalConstraint_site1Candidate,siteRelationalConstraint_site2DF,siteRelationalConstraint_site2Candidate)
                results[siteRelationalConstraint_constraintName] = euclideanDistance
                scores = siteRelationalConstraint.find("Scores")
                weight = scores.attrib['weight']
                totalWeight += float(weight)
                isZeroExclusionary = scores.attrib['isZeroExclusionary']
                default = scores.attrib['default']
                scoreStructure = []
                for scoreRow in scores:
                    lowerBoundInclusive = str(scoreRow.attrib['lowerBoundInclusive'])
                    upperBoundExclusive = str(scoreRow.attrib['upperBoundExclusive'])
                    score = str(scoreRow.attrib['score'])
                    scoreSet = [lowerBoundInclusive,upperBoundExclusive,score]
                    scoreStructure.append(scoreSet)
                evaluationDF =  pd.DataFrame(results,index=[0])
                evaluationDF = candidates.scoreDF(evaluationDF,siteRelationalConstraint_constraintName,scoreStructure,isZeroExclusionary=False)
                qafName = "%s_QAF" %(siteRelationalConstraint_constraintName)
                siteRelationalConstraint_constraintNames.append(qafName)
                results[qafName] = evaluationDF[qafName][0]
                results['MCDA_SCORE'] += float(weight) * float(evaluationDF[qafName][0])
        results['MCDA_SCORE'] /= float(totalWeight) # NOTE: MCDA NOT BEING USED
        scoreDF = pd.DataFrame(results,index=[0])
        scoreList = []
        for qafColumn in siteRelationalConstraint_constraintNames:
            scoreList.append(scoreDF[qafColumn][0])
        return scoreList

    except Exception as e:
        print e
        print "GA evaluate failed"

def testcode():
    i = 0
    for gene in individual:
        print "i = %s.  gene = %s" %(i,gene)
        candidate = listOfDataframes[i].iloc[gene,:]
        print candidate
        i += 1


    for siteRelationalConstraint in siteRelationalConstraints:
        if siteRelationalConstraint.tag == "SiteRelationalConstraint_Routing":
            print "Routing distance test"
            siteRelationalConstraint_constraintName = siteRelationalConstraint.attrib['constraintName']
            siteRelationalConstraint_candidate1TableIndex = int(siteRelationalConstraint.attrib['candidate1TableIndex'])
            siteRelationalConstraint_candidate1Index = int(siteRelationalConstraint.attrib['candidate1Index'])
            siteRelationalConstraint_candidate2TableIndex = int(siteRelationalConstraint.attrib['candidate2TableIndex'])
            siteRelationalConstraint_candidate2Index = int(siteRelationalConstraint.attrib['candidate2Index'])
            siteRelationalConstraint_note = siteRelationalConstraint.attrib['note']
            routingDistance = opt.evaluateCandidates_DrivingDistance(evaluationDFs[siteRelationalConstraint_candidate1TableIndex],siteRelationalConstraint_candidate1Index,evaluationDFs[siteRelationalConstraint_candidate2TableIndex],siteRelationalConstraint_candidate2Index)
            print routingDistance
        if siteRelationalConstraint.tag == "SiteRelationalConstraint_Euclidean":
            print "Euclidean distance test"
            siteRelationalConstraint_constraintName = siteRelationalConstraint.attrib['constraintName']
            siteRelationalConstraint_candidate1TableIndex = int(siteRelationalConstraint.attrib['candidate1TableIndex'])
            siteRelationalConstraint_candidate1Index = int(siteRelationalConstraint.attrib['candidate1Index'])
            siteRelationalConstraint_candidate2TableIndex = int(siteRelationalConstraint.attrib['candidate2TableIndex'])
            siteRelationalConstraint_candidate2Index = int(siteRelationalConstraint.attrib['candidate2Index'])
            siteRelationalConstraint_note = siteRelationalConstraint.attrib['note']
            euclideanDistance = opt.evaluateCandidates_EuclideanDistance(evaluationDFs[siteRelationalConstraint_candidate1TableIndex],siteRelationalConstraint_candidate1Index,evaluationDFs[siteRelationalConstraint_candidate2TableIndex],siteRelationalConstraint_candidate2Index)
            print euclideanDistance

    return 0



def evaluate_old(individual):
    try:
        # retreive geometries
        airfieldCandidate = airfieldDF
        airfieldCandidate.crs = {'init':'EPSG:3857'}
        airfieldCandidate = airfieldCandidate.to_crs({'init':'EPSG:4326'})
        airfieldCandidate = airfieldDF.iloc[individual[0],:]
        airfieldCentroidLon = float(airfieldCandidate['geometry'].centroid.x)
        airfieldCentroidLat = float(airfieldCandidate['geometry'].centroid.y)

        redCandidate = geomDF
        redCandidate.crs = {'init':'EPSG:3857'}
        redCandidate = redCandidate.to_crs({'init':'EPSG:4326'})
        redCandidate = geomDF.iloc[individual[1],:]
        redCentroidLon = float(redCandidate['geometry'].centroid.x)
        redCentroidLat = float(redCandidate['geometry'].centroid.y)

        blueCandidate = geom2DF
        blueCandidate.crs = {'init':'EPSG:3857'}
        blueCandidate = blueCandidate.to_crs({'init':'EPSG:4326'})
        blueCandidate = geom2DF.iloc[individual[2],:]
        blueCentroidLon = float(blueCandidate['geometry'].centroid.x)
        blueCentroidLat = float(blueCandidate['geometry'].centroid.y)

        # distance from airfield to site red
        redToAirfieldDistance = pgdissroute.calculateRouteDistance(airfieldCentroidLon,airfieldCentroidLat,redCentroidLon,redCentroidLat)
        # distance from airfield to site blue
        blueToAirfieldDistance = pgdissroute.calculateRouteDistance(airfieldCentroidLon,airfieldCentroidLat,blueCentroidLon,blueCentroidLat)

        # print for debugging
        print "red is %s m from airfield, at (%s,%s)" %(redToAirfieldDistance,redCentroidLon,redCentroidLat)
        print "blue is %s m from airfield, at (%s,%s)" %(blueToAirfieldDistance,blueCentroidLon,blueCentroidLat)
        return redToAirfieldDistance, blueToAirfieldDistance

    except Exception, e:
        print "Generated an error, returning very large number"
        return 999999,999999




def mutate(individual,listOfDataFrames,probElementMutation=0.1):
    attrDictionary = {}
    i = 0
    for df in listOfDataFrames:
        attrDictionary[i] = (0,len(df.index))
        i += 1
    for i in range(0,len(individual)):
        attr = individual[i]
        shouldWeMutate = np.random.random() <= probElementMutation
        if shouldWeMutate:
            newAttrLowerBound = attrDictionary[i][0]
            newAttrUpperBound = attrDictionary[i][1]
            newAttrValue = np.random.randint(newAttrLowerBound,newAttrUpperBound)
            individual[i] = newAttrValue
    return (individual,)

def createPopulation(populationSize,listOfDataFrames):
    genes = []
    dfIndex = 1
    for df in listOfDataFrames:
        dfName = "index%s" %(dfIndex)
        toolbox.register(dfName,random.randint,0,len(df.index))
        deapName = "toolbox.%s" %(dfName)
        genes.append(eval(deapName)) #This is HORRIBLE practice
        dfIndex += 1
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     genes, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    population = toolbox.population(n=populationSize)
    return population

def optimize_GA(listOfDataFrames,siteRelationalConstraints,popSize=10,nGenerations=10,
                pMutation=0.1,pCrossover=0.5,maxElite=4):
    try:
        # set up weights and fitness
        weights = [1.0 for i in range(0,len(siteRelationalConstraints))]
        creator.create("FitnessMax", base.Fitness, weights=weights)
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # set up the functions
        #toolbox = base.Toolbox #KLUDGE, resets the global
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", mutate)
        toolbox.register("select", tools.selNSGA2)

        # create the initial population
        population = createPopulation(popSize,listOfDataFrames)
        print population
        hallOfFame = tools.ParetoFront()
        logbook = tools.Logbook()

        # perform the GA
        stats = tools.Statistics(key= lambda ind: individual.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        for individual in population:
            individual.fitness.values = toolbox.evaluate(individual,listOfDataFrames,siteRelationalConstraints)
        record = stats.compile(population)
        hallOfFame.update(population)

        for generation in range(0,nGenerations):
            print "GENERATION %s" %(generation)
            theBestIndividuals = []
            if len(hallOfFame.items) <= maxElite:
                theBestIndividuals = hallOfFame.items
            elif len(hallOfFame.items) > 0:
                theBestIndividuals = hallOfFame.items[0:maxElite]
            reducedOffspring = toolbox.select(population, len(population)-len(theBestIndividuals))
            offspring = theBestIndividuals + reducedOffspring
           # Clone the selected individuals
            offspring = map(toolbox.clone, offspring)

            # Apply crossover on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= pCrossover:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Apply mutation on the offspring, with a probability assigned to each gene  # nopep8
            for mutant in offspring:
                toolbox.mutate(mutant,listOfDataFrames, pMutation)
                del mutant.fitness.values

             # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            for individual in invalid_ind:
                individual.fitness.values = toolbox.evaluate(individual,listOfDataFrames,siteRelationalConstraints)

            # The population is entirely replaced by the offspring
            population[:] = offspring
            hallOfFame.update(population)
            record=stats.compile(population)

            allScores = []
            for individual in population:
                allScores.append(individual.fitness.values)

            logbook.record(gen=generation, allIndividuals = list(population), allScores = allScores, top5 = theBestIndividuals[0], bestScore = theBestIndividuals[0].fitness.values,
                               **record)
            print theBestIndividuals[0], theBestIndividuals[0].fitness.values

        logbookDF = pd.DataFrame(logbook)
        return logbookDF

    except Exception as e:
        print e
        print "GA FAILED"

def cleanUp():
    weights=(-1.0,-1.0)
    creator.create("FitnessMin", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.FitnessMin)


    popSize = 10
    nGenerations = 10
    pMutation = 0.1
    pCrossover = 0.5
    maxElite = 4

    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selNSGA2)
    population = createPopulation(popSize)
    print population
    hallOfFame = tools.ParetoFront()
    logbook = tools.Logbook()

    # perform the GA
    stats = tools.Statistics(key= lambda ind: individual.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    for individual in population:
        individual.fitness.values = toolbox.evaluate(individual)
    record = stats.compile(population)
    hallOfFame.update(population)

    for generation in range(0,nGenerations):
        print "GENERATION %s" %(generation)
        theBestIndividuals = []
        if len(hallOfFame.items) <= maxElite:
            theBestIndividuals = hallOfFame.items
        elif len(hallOfFame.items) > 0:
            theBestIndividuals = hallOfFame.items[0:maxElite]
        reducedOffspring = toolbox.select(population, len(population)-len(theBestIndividuals))
        offspring = theBestIndividuals + reducedOffspring
       # Clone the selected individuals
        offspring = map(toolbox.clone, offspring)

        # Apply crossover on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= pCrossover:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation on the offspring, with a probability assigned to each gene  # nopep8
        for mutant in offspring:
            toolbox.mutate(mutant, pMutation)
            del mutant.fitness.values

         # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for individual in invalid_ind:
            individual.fitness.values = toolbox.evaluate(individual)

        # The population is entirely replaced by the offspring
        population[:] = offspring
        hallOfFame.update(population)
        record=stats.compile(population)

        allScores = []
        for individual in population:
            allScores.append(individual.fitness.values)

        logbook.record(gen=generation, allIndividuals = list(population), allScores = allScores, top5 = theBestIndividuals[0], bestScore = theBestIndividuals[0].fitness.values,
                           **record)
        print theBestIndividuals[0], theBestIndividuals[0].fitness.values

    logbookDF = pd.DataFrame(logbook)

    logbookDF

def convertLogBookIntoGenerationalCoordinates(logbookDF):
    airfieldDF = airfieldCandidates
    redDF = geomDF
    redDF.crs = {'init':'EPSG:4326'}
    blueDF = geom2DF
    blueDF.crs = {'init':'EPSG:4326'}
    airfieldDF.crs = {'init':'EPSG:3857'}
    airfieldDF = airfieldDF.to_crs({'init':'EPSG:4326'})
    heat_data = []
    # each row is one generation
    for i,row in logbookDF.iterrows():
        combinedLocations = []
        airfields = []
        reds = []
        blues = []
        rowInd = row['allIndividuals']
        for individual in rowInd:
            airfieldIndex = individual[0]
            redIndex = individual[1]
            blueIndex = individual[2]

            airfieldCandidate = airfieldDF.iloc[airfieldIndex,:].geometry
            airfieldCandidateCentroid = airfieldCandidate.centroid
            airfields.append([airfieldCandidateCentroid.y,airfieldCandidateCentroid.x])
            combinedLocations.append([airfieldCandidateCentroid.y,airfieldCandidateCentroid.x])

            redCandidate = redDF.iloc[redIndex,:].geometry
            redCandidateCentroid = redCandidate.centroid
            reds.append([redCandidateCentroid.y,redCandidateCentroid.x])
            combinedLocations.append([redCandidateCentroid.y,redCandidateCentroid.x])

            blueCandidate = blueDF.iloc[blueIndex,:].geometry
            blueCandidateCentroid = blueCandidate.centroid
            blues.append([blueCandidateCentroid.y,blueCandidateCentroid.x])
            combinedLocations.append([blueCandidateCentroid.y,blueCandidateCentroid.x])
        heat_data.append(combinedLocations)
    return heat_data

def generateHeatMap(logbookDF,mapPath):
    heat_data = convertLogBookIntoGenerationalCoordinates(logbookDF)
    m = io.Map()
    m.addTimeSeriesHeatMapFromArray(heat_data)
    m.saveMap(mapPath)
    m.map
