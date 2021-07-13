import sys
import csv
import signal
import argparse
from run_single_instance import runSingleInstanceForAllCategory


'Main function'

#Commandline arguments processing

# Instantiate the parser
parser = argparse.ArgumentParser(description='FFN run all instances description')

# Optional benchmark category
parser.add_argument('--category',
                    help='An optional benchmark category')

# Optional report filepath
parser.add_argument('--reportfile',
                    help='An optional report file path')

args = parser.parse_args()
category = args.category

#Set default categery 
if ( category is None ):
    category = "test"
    print ("\n!!! No benchmark category is provided on the command line!")
    print ("Default benchmark category is taken as - \"test\"")

reportFile = args.reportfile

#Set default reportFile path
if ( reportFile is None ):
    reportFile ="report_"+category+".txt"
    print ("\n!!! No result_file is provided on the command line!")
    print (" Taking default result_file -\"{0}\"".format(reportFile))


#Reading cat_instance.csv for .onnx file path, .vnnlib file path and timeout

catInsCsvFile = "benchmarks/"+category+"/"+category+"_instances.csv"
insCsvFile = open(catInsCsvFile, 'r')
outFile = open(reportFile, 'w')
reader = csv.reader(insCsvFile)
for row in reader:
    onnxFile = "benchmarks/"+category+"/"+row[0]
    if (onnxFile.endswith('.onnx') == False):
       print("\n!!!Wrong onnx file format for -\"{0}\"".format(onnxFile))
       continue
    vnnlibFile = "benchmarks/"+category+"/"+row[1]
    if (vnnlibFile.endswith('.vnnlib') == False):
       print("\n!!!Wrong vnnlib file format for -\"{0}\"".format(vnnlibFile))
       continue
    timeout = row[2]

    resultStr = runSingleInstanceForAllCategory(onnxFile,vnnlibFile,"out.txt",timeout)
    if (not resultStr):
       resultStr = "timeout,"+timeout
    printStr=onnxFile+","+vnnlibFile+","+resultStr+"\n"
    outFile.write(printStr)

insCsvFile.close()
outFile.close()
