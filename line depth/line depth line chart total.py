import json
import matplotlib.pyplot as plt
from collections import Counter


#NOTES FOR LINE DEPTH
"""
Depth
"bugLineNum" : The line in which the bug exists in the buggy version of the file.
"bugNodeStartChar" : The character index (i.e., the number of characters in the java file that must be read before encountering the first one of the AST node) at which the affected ASTNode starts in the buggy version of the file.
"bugNodeLength" : The length of the affected ASTNode in the buggy version of the file.
"fixLineNum" : The line in which the bug was fixed in the fixed version of the file.
"fixNodeStartChar" : The character index (i.e., the number of characters in the java file that must be read before encountering the first one of the AST node) at which the affected ASTNode starts in the fixed version of the file.
"fixNodeLength" : The length of the affected ASTNode in the fixed version of the file.

TLDR!!!

"bugLineNum" : which line is the bug on
"bugNodeStartChar" : how many characters are before the bug starts in the file
"bugNodeLength" : how long is the bugged code
"fixLineNum" : which line is the bug fix on
"fixNodeStartChar" : how many characters are before the bug fix starts in the file
"fixNodeLength" : how long is the bug fix code


"""

#WHAT DATA DO YOU WANT TO COLLECT FROM THE LIST ABOVE
datafeature = "bugLineNum"


#README: to use this you'll need to change the location of the path for the extracted stubbs set
#ensure you have the modules above installed (json matplotlib collections)
#set bugnameplot to whatever pie chart you want to see



#path for location of extracted json file
with open ('sstubs-0104.json/sstubs-0104.json', encoding="utf8") as f:
    data = json.load(f)

#print(data[0])

#list of bug types in order
bugtypelist = []
#list of bug locational data in order
buglocation = []

#fetch data
for x in range(len(data)):
    bugtypelist.append(data[x]["bugType"])
    buglocation.append(data[x][datafeature])
    

#add to dict values
dataanalysis = {}
for z in range(len(bugtypelist)):
#for z in range(5):
    if bugtypelist[z] not in dataanalysis:
        dataanalysis[bugtypelist[z]] = [buglocation[z]]
    if bugtypelist[z] in dataanalysis:
        #print(dataanalysis)
        dataanalysis[bugtypelist[z]].append(buglocation[z])
    
        #print(dataanalysis[bugtypelist[z]])
    

#datanalysis has the format 'CHANGE_OPERATOR': [11, 11, 6, 9] etc
#to parse data from data analysis use
#list(Counter(dataanalysis['CHANGE_OPERATOR']).keys())


#sns.set_style("whitegrid")
#SET THE BUG YOU WANT TO SEE THE  CHART FOR HERE
bugnameplot = 'CHANGE_MODIFIER'

#plot
"""
#PLOT FOR PIE CHART
pie, ax = plt.subplots(figsize=[10,6])
Labels = [k for k in Counter(dataanalysis[bugnameplot]).keys()]
Data   = [float(v) for v in Counter(dataanalysis[bugnameplot]).values()]
plt.pie(x = Data, labels=Labels, autopct="%.1f%%", pctdistance=0.5);
plt.title("Layers of directories for " + bugnameplot, fontsize=14);
"""

#PLOT FOR single HISTOGRAM

#xvalue = list(Counter(dataanalysis[bugnameplot]).keys())

#yvalue = list(Counter(dataanalysis[bugnameplot]).values())

#ysortedvalues = [m for _,m in sorted(zip(xvalue,yvalue))]

#xsortedvalues = sorted(xvalue)

histvalues = dataanalysis[bugnameplot]


plt.hist(histvalues, bins = 20)

plt.title('Line Depth Histogram for ' + bugnameplot, fontsize=14)
plt.xlabel('Line Depth')
plt.ylabel('Frequency')



#PLOT FOR all LINE GRAPHS
"""
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

"""



plt.show()


