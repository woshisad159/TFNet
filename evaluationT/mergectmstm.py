#!/usr/bin/env python

import sys

ctmFile=sys.argv[1]
stmFile=sys.argv[2]

ctm=open(ctmFile,"r")
stm=open(stmFile,"r")

ctmDict=[]
stmDict=[]
addedlines=0

ctmToOrder={}

# read in ctm so we can order it
for idx,line in enumerate(ctm):   
    l=line.strip().split()
    corpusid=l[0]
    if corpusid not in ctmToOrder:
        ctmToOrder[corpusid]=[]
    ctmToOrder[corpusid].append(l)

#read in stm
for idx,line in enumerate(stm):
    l=line.strip().split()
    stmDict.append(l)
      
stm.close()
ctm.close()

for stmline in stmDict: #follow the stm order
    corpusid=stmline[0]
    for l in ctmToOrder[corpusid]:
        ctmDict.append(l)

#no everything is sorted and we can proceed
stm=open(stmFile,"r")

for idx,line in enumerate(stm):
    l=line.strip().split()
    stmDict.append(l)

    if len(ctmDict) > idx+addedlines and ctmDict[idx+addedlines][0]==l[0]: #ctm and stm match:
        if len(ctmDict)>idx+addedlines+1:
            while (len(ctmDict)>idx+addedlines+1) and (ctmDict[idx+addedlines+1][0]==l[0]):
                addedlines+=1
    else:
        ctmDict.insert(idx+addedlines,[l[0],"1 0.000 0.030 [EMPTY]"])

stm.close()
ctm=open(ctmFile,"w")
for l in ctmDict:
    ctm.write(" ".join(l)+"\n")
ctm.close()



