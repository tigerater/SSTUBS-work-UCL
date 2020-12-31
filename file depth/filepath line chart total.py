import json
import matplotlib.pyplot as plt
from collections import Counter


#README: to use this you'll need to change the location of the path for the extracted stubbs set
#ensure you have the modules above installed (json matplotlib collections)
#set bugnameplot to whatever pie chart you want to see



#path for location of extracted json file
with open ('sstubsLarge-0104.json', encoding="utf8") as f:
    data = json.load(f)

#print(data[0])

#list of bug types in order
bugtypelist = []
#list of bug file paths in order
bugfilepath = []

#fetch data
for x in range(len(data)):
    bugtypelist.append(data[x]["bugType"])
    bugfilepath.append(data[x]["bugFilePath"])

#count directory depth (via file path)
bugfilepathSlash = []
for y in range(len(bugtypelist)):
    bugfilepathSlash.append(bugfilepath[y].count("/"))
    

#add to dict values
dataanalysis = {}
for z in range(len(bugtypelist)):
#for z in range(5):
    if bugtypelist[z] not in dataanalysis:
        dataanalysis[bugtypelist[z]] = [bugfilepathSlash[z]]
    if bugtypelist[z] in dataanalysis:
        #print(dataanalysis)
        dataanalysis[bugtypelist[z]].append(bugfilepathSlash[z])
    
        #print(dataanalysis[bugtypelist[z]])
    

#datanalysis has the format 'CHANGE_OPERATOR': [11, 11, 6, 9] etc
#to parse data from data analysis use
#list(Counter(dataanalysis['CHANGE_OPERATOR']).keys())


#sns.set_style("whitegrid")
#SET THE BUG YOU WANT TO SEE THE PIE CHART FOR HERE
bugnameplot = 'DELETE_THROWS_EXCEPTION'

#plot
"""
#PLOT FOR PIE CHART
pie, ax = plt.subplots(figsize=[10,6])
Labels = [k for k in Counter(dataanalysis[bugnameplot]).keys()]
Data   = [float(v) for v in Counter(dataanalysis[bugnameplot]).values()]
plt.pie(x = Data, labels=Labels, autopct="%.1f%%", pctdistance=0.5);
plt.title("Layers of directories for " + bugnameplot, fontsize=14);
"""

#PLOT FOR single LINE GRAPH
"""
xvalue = list(Counter(dataanalysis['CHANGE_OPERATOR']).keys())
yvalue = list(Counter(dataanalysis['CHANGE_OPERATOR']).values())

ysortedvalues = [m for _,m in sorted(zip(xvalue,yvalue))]

xsortedvalues = sorted(xvalue)


plt.plot(xsortedvalues, ysortedvalues, label = "line 1")

plt.title('Layers of directories for ' + bugnameplot, fontsize=14)
plt.xlabel('File Depth (calculated by number of "/")')
plt.ylabel('Frequency')
"""


#PLOT FOR all LINE GRAPHS

for n in list(dataanalysis.keys()):
    xvalue = list(Counter(dataanalysis[n]).keys())
    yvalue = list(Counter(dataanalysis[n]).values())

    ysortedvalues = [m for _,m in sorted(zip(xvalue,yvalue))]

    xsortedvalues = sorted(xvalue)


    plt.plot(xsortedvalues, ysortedvalues, label = n)

plt.title('Layers of directories for all errors', fontsize=14)
plt.legend()
plt.xlabel('File Depth (calculated by number of "/")')
plt.ylabel('Frequency')





plt.show()


