# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 16:34:01 2019

@author: matthew
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import glob as g
from collections import OrderedDict
           
def read_month(fname,):
    with open(fname) as dmonth:
        monthreader = csv.reader(dmonth, delimiter=',')
        dates = next(monthreader)
        enddate = dates.index('')
        monthdata = [[t.lower() for t in timeslot[1:enddate]] for timeslot in monthreader]
        return np.asarray(monthdata)

def binweek(x):
    """Bin x (365,1) into weeks, (last week 8 days)
    """
    xcut = np.copy(x[:-1])
    xsplit = np.split(xcut, 52)
    xweeks = [np.sum(week) for week in xsplit]
    xweeks[-1] += x[-1]
    return np.asarray(xweeks)


activity = {'sleep': 0,
            'phd work': 1,
            'phd admin': 2,
            'life admin': 3,
            'outreach': 4,
            'soc rec': 5,
            'soc rel': 6,
            'family time': 7,
            'projects': 8,
            'reading': 9,
            'tv': 10,
            'relaxing': 11,
            'exercise': 12,
            'hygiene': 13,
            'housework': 14,
            'food': 15,
            'qt': 16,
            'travel': 17,
            'golf': 18,
            'board games': 19,
            'd&d': 20,
            'video games': 21,
            'shopping': 22,
            'misc rec': 23}
"""
try:
    wholeyear = np.genfromtxt('year.dat', delimiter=',', dtype='str')
except OSError:
    mnths = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    #months.remove('year.csv')
    monthdata = {}
    for m in mnths:
        monthdata[m] = read_month(m+'.csv')
    
    #alldata = 
    wholeyear = np.concatenate(([monthdata[md] for md in mnths]), axis=0)
    np.savetxt('year.dat', wholeyear, delimiter=',', fmt='%s')
"""

with open('allyear.csv', 'r') as fin:
    yearreader = csv.reader(fin, delimiter=',')
    allyear = [[y.lower() for y in x] for x in yearreader]
wholeyear = np.asarray(allyear)

#wholeyear = np.genfromtxt('allyear.csv', delimiter=',', dtype='str').lower()

ToD = []
for i in range(24):
    for j in range(3):
        ToD.append('{:02.0f}:{:02.0f}'.format(i, j*20))

"""Sleeping"""

bedtimes = []
alarms = []
nights = []
for d in range(365):
    bedtime = np.min(np.argwhere(wholeyear[:, d] == 'sleep'))
    alarm = np.min(np.argwhere(wholeyear[bedtime:, d] != 'sleep')) + bedtime
#    if bedtime > 15:
#        alarm = np.argwhere(wholeyear[bedtime:, d] != 'sleep').min()+bedtime
#    else:
#        alarm = np.argwhere(wholeyear[12:, d] != 'sleep').min()+12
    alarms.append(alarm)
    if bedtime == 0:
        try:
            altbedtime = np.argwhere(wholeyear[-12:, d-1] == 'sleep').min() + 60
            bedtimes.append(altbedtime)
        except ValueError:
            bedtimes.append(0)
    else: bedtimes.append(bedtime)
    nights.append(alarm-bedtime)

bedscience = []
for b in bedtimes:
    if b > 54:
        bedscience.append(b-72)
    else:
        bedscience.append(b)
"""
plt.figure()
plt.hist(bedscience, np.arange(-12., 59., 1), alpha=0.8, label='First asleep')
plt.hist(alarms, np.arange(-12., 59., 1), alpha=0.8, label='First awake')
a = plt.gca()
a.set_xticklabels([])
plt.legend()
plt.xticks(range(-12, 55, 6), ToD[-12::6]+ToD[:55:6])
plt.xlabel('Time of Day')
plt.ylabel('Occurences per year')
plt.show(block=False)

nights = np.array(nights, dtype='float')
nights *= 1/3

plt.figure()
plt.hist(nights, np.arange(0, 12, 1/3), alpha=0.8)
plt.xlabel('Duration of Sleep [Hrs]')
plt.ylabel('Occurences per year')
plt.show()
"""


"""Work Duration"""
phdw = []
phda = []
workstart = []
workend = []
for d in range(365):
    phdw.append(np.sum(wholeyear[:, d]=='phd work'))
    phda.append(np.sum(wholeyear[:, d]=='phd admin'))
    failure = 999
    #try:
    if ('phd work' in wholeyear[alarms[d]:,d]) or ('phd admin' in wholeyear[alarms[d]:,d]):
        try:
            ws1 = np.min(np.argwhere(wholeyear[alarms[d]:, d]=='phd work')+alarms[d])
        except ValueError:
            ws1 = failure
        try:
            ws2 = np.min(np.argwhere(wholeyear[alarms[d]:, d]=='phd admin')+alarms[d])
        except ValueError:
            ws2 = failure
        """if ws1 == 999 and ws2 == 999:
            try:
                ws3 = np.min(np.argwhere(wholeyear[:bedtimes[d+1], d+1]=='phd work')) + 72
            except ValueError:
                ws3 = 999
            try:
                ws4 = np.min(np.argwhere(wholeyear[:bedtimes[d+1], d+1]=='phd admin')) + 72
            except ValueError:
                ws4 = 999
        else:
            ws3 = 999
            ws4 = 999
        """
        ws = np.min([ws1, ws2])#, ws3, ws4)
            
        try:
            we1 = np.max(np.argwhere(wholeyear[alarms[d]:, d]=='phd work')+alarms[d])
        except ValueError:
            we1 = -1
        try:
            we2 =  np.max(np.argwhere(wholeyear[alarms[d]:, d]=='phd admin')+alarms[d])
        except ValueError:
            we2 = -1
        if bedtimes[d+1] < 60:
            try:
                we3 = np.max(np.argwhere(wholeyear[:bedtimes[d+1], d+1]=='phd work')) + 72
            except ValueError:
                we3 = -1
            try:
                we4 = np.max(np.argwhere(wholeyear[:bedtimes[d+1], d+1]=='phd admin')) + 72
            except ValueError:
                we4 = -1
        else:
            we3 = -1
            we4 = -1
        we = np.max([we1, we2, we3, we4])
        if we == -1:
            print('No work end found for {}'.format(d))
                    
        workstart.append(ws)
        workend.append(we)
    else:
        workstart.append(-2)
        workend.append(-2)
    #except ValueError:
    #    workstart.append(np.nan)
    #    workend.append(np.nan)

totwork = (np.asarray(phdw) + np.asarray(phda))/3

workspan = []
for s,e in zip(workstart, workend):
    if e < 0 and s < 0:
        workspan.append(0)
    elif e>s:
        workspan.append((1+e-s))
    elif e==s:
        workspan.append(1)
    else:
        print('Not sure {}{}'.format(s, e))
        workspan.append(0)
"""
plt.figure()
plt.hist(totwork, np.arange(-1/6, 80/3, 1/3))
plt.xlabel('Duration of PhD Work [Hrs]')
plt.ylabel('Occurences per year')
plt.xlim(-1/6,79/3)
plt.show()

wk_pw = binweek(phdw)/3
wk_pa = binweek(phda)/3
plt.figure()
plt.bar(range(52), wk_pw, color='midnightblue', width=1, label='Work')
plt.bar(range(52), wk_pa, bottom=wk_pw, color='steelblue', width=1, label='Admin')
plt.xlabel('Week of Year')
plt.ylabel('Hours per Week')
plt.xlim(-0.5,52.5)
plt.show()

wk_sleep = binweek(nights)/3
wk_work = binweek(np.asarray(phdw) + np.asarray(phda))/3
pfit = np.polyfit(wk_sleep, wk_work, 1)
tsleep = np.arange(44, 68, 2)
sw_fit = pfit[1] + tsleep*pfit[0]
plt.figure()
plt.scatter(wk_sleep, wk_work)
plt.plot(tsleep, sw_fit)
plt.xlabel('Hours Sleep per Week')
plt.ylabel('Hours Work per Week')
plt.show()

workspan = np.asarray(workspan)/3
plt.figure()
plt.bar(range(365), workspan, width=1)
plt.xlabel('Day of year')
plt.ylabel('Span of working Hours')
plt.xlim(-0.5,365.5)
plt.show(block=False)

plt.figure()
plt.hist(workspan, np.arange(-1/6, 120/6, 1/3))
plt.xlabel('Span of working Hours')
plt.ylabel('Occurences per year')
plt.xlim(-1/6,119/6)
plt.show(block=False)
"""








