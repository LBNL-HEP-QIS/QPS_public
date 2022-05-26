import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib
import math

import MakeObservables as mo

def ptype(x):
    #parses particle type
    if x=='000':
        return '0'    
    if x=='001':
        return 'phi'   
    if x=='100':
        return 'f1'   
    if x=='101':
        return 'f2'   
    if x=='110':
        return 'af1'   
    if x=='111':
        return 'af2'   
    else:
        return "NAN"


def bar_plot2(counts, events, eps, g1, g2, counts2= None, save=True, wReg=True):

    width= 0.2
    firstisf1_y = np.zeros(6)
    firstisf1_ey = np.zeros(6)
    firstisf1_x = np.arange(1 - width, 7 - width, 1)

    firstisf2_y = np.zeros(6)
    firstisf2_ey = np.zeros(6)
    firstisf2_x = np.arange(1, 7, 1)

    if counts2 != None:
        firstisf1b_y = np.zeros(6)
        firstisf1b_ey = np.zeros(6)
        firstisf1b_x = np.arange(1 + width, 7 + width, 1)

        firstisf2b_y = np.zeros(6)
        firstisf2b_ey = np.zeros(6)

    mymap = {}
    mymap['0','0']=1
    mymap['phi','0']=2
    mymap['0','phi']=2
    mymap['phi','phi']=3
    mymap['af1','f1']=4
    mymap['f1','af1']=4
    mymap['af2','f2']=5
    mymap['f2','af2']=5
    mymap['af2','f1']=6
    mymap['f2','af1']=6
    mymap['af1','f2']=6
    mymap['f1','af2']=6

    mycounter = 0
    for c in counts:
        print(mycounter, c, ptype(c.split()[-3 - wReg]), ptype(c.split()[-2 - wReg]), ptype(c.split()[-1 - wReg]), counts[c])
        mycounter+=1

        x= mymap[ptype(c.split()[-3 - wReg]), ptype(c.split()[-2 - wReg])] - 1
        if (ptype(c.split()[-1 - wReg])=='f1'):
            firstisf1_y[x]+= 100*counts[c]/events
            firstisf1_ey[x]+= 100*counts[c]**0.5/events
            pass
        if (ptype(c.split()[-1 - wReg])=='f2'):
            firstisf2_y[x]+= 100*counts[c]/events
            firstisf2_ey[x]+= 100*counts[c]**0.5/events
            pass

    if counts2 != None:
        for c in counts2:
            x= mymap[ptype(c.split()[-3 - wReg]), ptype(c.split()[-2 - wReg])] - 1
            if (ptype(c.split()[-1 - wReg])=='f1'):
                firstisf1b_y[x]+= 100*counts2[c]/events
                firstisf1b_ey[x]+= 100*counts2[c]**0.5/events
                #firstisf1b_x+=[0.2+mymap[ptype(c.split()[-4]),ptype(c.split()[-3])]]
                pass
            if (ptype(c.split()[-1 - wReg])=='f2'):
                firstisf2b_y[x]+= 100*counts2[c]/events
                firstisf2b_ey[x]+= 100*counts2[c]**0.5/events
                pass



    f = plt.figure(figsize=(10, 8))
    ax = f.add_subplot(1, 1, 1)
    plt.ylim((100*1e-4, 100*5.))
    plt.xlim((1 - 3*width, 6 + 2*width))
    ax.set_yscale("log", nonposy='clip')
    ax.set_ylabel('Probability [%]', size=20)
    bar1 = plt.bar(firstisf1_x, firstisf1_y, color='#228b22', width=width, label=r"$f' = f_{1}, g_{12}= 1$", hatch='\\') #,yerr=firstisf1_ey)
    bar1b = plt.bar(firstisf2_x, firstisf2_y, color='#9AEE9A', width=width, label=r"$f' = f_{2}, g_{12}= 1$", hatch='//') #,yerr=firstisf2_ey)

    ax.set_xticks(np.arange(1, 7, 1))
    ax.set_xticklabels( (r"$f_{1}\rightarrow f'$", r"$f_{1}\rightarrow f'\phi$", r"$f_{1}\rightarrow f'\phi\phi$",r"$f_{1}\rightarrow f' f_{1} \bar{f}_{1}$",r"$f_{1}\rightarrow f' f_{2} \bar{f}_{2}$",r"$f_{1}\rightarrow f' f_{1/2} \bar{f}_{2/1}$") )

    if counts2 != None:
        bar2 = plt.bar(firstisf1b_x, firstisf1b_y, color='#01B9FF', width=width, label=r"$f' = f_{1}, g_{12}= 0$", alpha=1.0) #,hatch="//")
        bar2b = plt.bar(firstisf1b_x, firstisf2b_y, color='#01B9FF', width=width, label=r"$f' = f_{2}, g_{12}= 0$", alpha=1.0) #,hatch="//")
        pass
    else:
        pass

    plt.legend(loc='upper right', prop={'size': 14})
    plt.title(r"2-step Full Quantum Simulation", fontsize=24)
    plt.text(0.6, 220, r"$(g_{1},g_{2},\epsilon) = ("+str(g1)+","+str(g2)+",10^{-3})$", fontsize=16)

    if save == True:
        f.savefig("sim2step_states_shots=%d.pdf" %(events))
    plt.show()







def bar_plot3(counts, events, eps, g1, g2, initialParticles, counts2= None, save=True):

    width= 0.2
    y_f1 = np.zeros(10)
    ey_f1 = np.zeros(10)
    x_f1 = np.arange(1 - width, 11 - width, 1)

    y_f2 = np.zeros(10)
    ey_f2 = np.zeros(10)
    x_f2 = np.arange(1, 11, 1)

    if counts2 != None:
        y2_f1 = np.zeros(10)
        ey2_f1 = np.zeros(10)
        x2_f1 = np.arange(1 + width, 11 + width, 1)

        #y2_f2 = np.zeros(10)
        #ey2_f2 = np.zeros(10)
        #x2_f2 = []


    mymap = {}
    # 0
    mymap['0', '0', '0']= 1

    # phi
    mymap['phi', '0', '0']= 2
    mymap['0', 'phi', '0']= 2
    mymap['0', '0', 'phi']= 2

    # phi phi
    mymap['phi', 'phi', '0']= 3
    mymap['phi', '0', 'phi']= 3
    mymap['0', 'phi', 'phi']= 3
    
    # phi phi phi
    mymap['phi', 'phi', 'phi']= 4

    # f1 af1
    mymap['af1', 'f1', '0']= 5
    mymap['f1', 'af1', '0']= 5
    mymap['af1', '0', 'f1']= 5
    mymap['f1', '0', 'af1']= 5
    mymap['0', 'af1', 'f1']= 5
    mymap['0', 'f1', 'af1']= 5

    # f2 af2
    mymap['af2', 'f2', '0']= 6
    mymap['f2', 'af2', '0']= 6
    mymap['af2', '0', 'f2']= 6
    mymap['f2', '0', 'af2']= 6
    mymap['0', 'af2', 'f2']= 6
    mymap['0', 'f2', 'af2']= 6

    # f1 af2 / f2 af1
    mymap['af2', 'f1', '0']= 7
    mymap['f2', 'af1', '0']= 7
    mymap['af1', 'f2', '0']= 7
    mymap['f1', 'af2', '0']= 7

    mymap['af2', '0', 'f1']= 7
    mymap['f2', '0', 'af1']= 7
    mymap['af1', '0', 'f2']= 7
    mymap['f1', '0', 'af2']= 7

    mymap['0', 'af2', 'f1']= 7
    mymap['0', 'f2', 'af1']= 7
    mymap['0', 'af1', 'f2']= 7
    mymap['0', 'f1', 'af2']= 7

    # f1 af1 phi
    mymap['af1', 'f1', 'phi']= 8
    mymap['f1', 'af1', 'phi']= 8
    mymap['af1', 'phi', 'f1']= 8
    mymap['f1', 'phi', 'af1']= 8
    mymap['phi', 'af1', 'f1']= 8
    mymap['phi', 'f1', 'af1']= 8

    # f2 af2 phi
    mymap['af2', 'f2', 'phi']= 9
    mymap['f2', 'af2', 'phi']= 9
    mymap['af2', 'phi', 'f2']= 9
    mymap['f2', 'phi', 'af2']= 9
    mymap['phi', 'af2', 'f2']= 9
    mymap['phi', 'f2', 'af2']= 9

    # (f1 af2 / f2 af1) phi
    mymap['af2', 'f1', 'phi']= 10
    mymap['f2', 'af1', 'phi']= 10
    mymap['af1', 'f2', 'phi']= 10
    mymap['f1', 'af2', 'phi']= 10

    mymap['af2', 'phi', 'f1']= 10
    mymap['f2', 'phi', 'af1']= 10
    mymap['af1', 'phi', 'f2']= 10
    mymap['f1', 'phi', 'af2']= 10

    mymap['phi', 'af2', 'f1']= 10
    mymap['phi', 'f2', 'af1']= 10
    mymap['phi', 'af1', 'f2']= 10
    mymap['phi', 'f1', 'af2']= 10


    #mycounter = 0
    for c in counts:
        x= mymap[ptype(c.split()[5]), ptype(c.split()[6]), ptype(c.split()[7])] - 1 # - 1 for zero-indexing
        pList= list((ptype(c.split()[5]), ptype(c.split()[6]), ptype(c.split()[7]), ptype(c.split()[8])))
        #print(mycounter, c, pList, counts[c])
        if (ptype(c.split()[-2])=='f1'):
            y_f1[x]+= 100*counts[c]/events
            ey_f1[x]+= 100*counts[c]**0.5/events
            #x_f1+= [-1.5*offset + 6*mymap[ptype(c.split()[5]), ptype(c.split()[6]), ptype(c.split()[7])]]
            pass
        if (ptype(c.split()[-2])=='f2'):
            y_f2[x]+= 100*counts[c]/events
            ey_f2[x]+= 100*counts[c]**0.5/events
            #x_f2+= [-0.5*offset + 6*mymap[ptype(c.split()[5]), ptype(c.split()[6]), ptype(c.split()[7])]]
            pass
        pass            

    if counts2 != 0:
        for c in counts2:
            x= mymap[ptype(c.split()[5]), ptype(c.split()[6]), ptype(c.split()[7])] - 1
            pList= list((ptype(c.split()[5]), ptype(c.split()[6]), ptype(c.split()[7]), ptype(c.split()[8])))
            #print(mycounter, c, pList, counts[c])
            if (ptype(c.split()[-2])=='f1'):
                y2_f1[x]+= 100*counts2[c]/events
                ey2_f1[x]+= 100*counts2[c]**0.5/events
                #x2_f1+= [0.5*offset + 6*mymap[ptype(c.split()[5]), ptype(c.split()[6]), ptype(c.split()[7])]]
                pass
            #if (ptype(c.split()[-2])=='f2'):
                #y2_f2+= [100*counts2[c]/events]
                #ey2_f2+= [100*counts2[c]**0.5/events]
                #x2_f2+= [1.5*offset + 6*mymap[ptype(c.split()[5]), ptype(c.split()[6]), ptype(c.split()[7])]]
                #pass
            pass


    f = plt.figure(figsize=(14, 10))
    ax = f.add_subplot(1, 1, 1)
    plt.ylim((100*1e-4, 100*5.))
    plt.xlim((1 - 3*width, 10 + 2*width))
    ax.set_yscale("log", nonposy='clip')
    ax.set_ylabel('Probability [%]', size= 24)
    bar1_f1 = plt.bar(x_f1, y_f1, color='#228b22', width=width, label=r"$g_{12}= 1, f'= f_{1}$", hatch='//') #,yerr=firstisf1_ey)
    bar1_f2 = plt.bar(x_f2, y_f2, color='#9AEE9A', width=width, label=r"$g_{12}= 1, f'= f_{2}$", hatch='\\') #,yerr=firstisf1_ey)
    #plt.hist(x_f1, weights=y_f1, bins= np.arange(6-1.5*offset, 72-1.5*offset, 6), align= 'mid', color='#228b22', width=offset, label=r"$f' = f_{1}, g_{12}= 1$", hatch='//')
    #plt.hist(x_f2, weights=y_f2, bins= np.arange(6-0.5*offset, 72-0.5*offset, 6), align= 'mid', color='#9AEE9A', width=offset, label=r"$f' = f_{2}, g_{12}= 1$", hatch='\\')

    if counts2!= 0:
        bar2_f1 = plt.bar(x2_f1, y2_f1, color='#01B9FF', width=width, label=r"$g_{12}= 0, f'= f_{1}$")
        #bar2_f2 = plt.bar(x2_f2, y2_f2, color='#C0EDFE', width=offset, label=r"$g_{12}= 0, f'= f_{2}$")
        #plt.hist(x2_f1, weights=y2_f1, bins= np.arange(6+0.5*offset, 72+0.5*offset, 6), align= 'mid', color='#01B9FF', width=offset, label=r"$f' = f_{1}, g_{12}= 0$")
        #leg2 = ax.legend([bar1_f1, bar1_f2],[r'$f = f_{a}$',r'$f = f_{b}$'], loc='upper right', frameon=False, prop={'size': 12.5}, bbox_to_anchor=(1.,0.8))

    pmap= {'f1': r'$f_{1}$', 'af1': r'$f_{1}$', 'f2': r'$f_{2}$', 'af2': r'$f_{2}$', 'phi': r'$\phi$'}
    iP_str= ''
    for j in range(len(initialParticles)):
        iP_str+= pmap[ptype(initialParticles[j])]
        if j > 0: iP_str+= ', '

    ax.set_xticks(np.arange(1, 11, 1))
    ax.set_xticklabels((iP_str + r"$\rightarrow f'$", iP_str + r"$\rightarrow f'\phi$", iP_str + r"$\rightarrow f'\phi\phi$", iP_str + r"$\rightarrow f'\phi\phi\phi$",
                        iP_str + r"$\rightarrow f' f_{1} \bar{f}_{1}$", iP_str + r"$\rightarrow f' f_{2} \bar{f}_{2}$", iP_str + r"$\rightarrow f' f_{1/2} \bar{f}_{2/1}$",
                        iP_str + r"$\rightarrow f' f_{1} \bar{f}_{1} \phi$", iP_str + r"$\rightarrow f' f_{2} \bar{f}_{2} \phi$", iP_str + r"$\rightarrow f' f_{1/2} \bar{f}_{2/1} \phi$"), size= 10)
    plt.xlabel('Final State', size=24)

    plt.legend(loc='upper right',prop={'size': 20})



    #plt.text(-0.3, 55*4, r"3-step Full Quantum Simulation", fontsize=24)
    plt.title(r"3-step Full Quantum Simulation", fontsize=28)
    plt.text(2.8, 200, r"$(g_{1},g_{2},\epsilon) = ("+str(g1)+","+str(g2)+",10^{-3})$", fontsize=20)

    plt.text(2.8, 100, r"Initial: " + iP_str, fontsize=20)

    if save == True:
        f.savefig("sim3step_states_shots=%d.pdf" %(events))
    plt.show()




def bar_plot_numbers(counts, events, eps, g1, g2, N, ni, counts2= None):
    # Plots counts vs. number configuration

    # Can handle two different couplings, counts and counts2

    # For now assume initially start with only fermions

    mycounter = 0

    plot1_y = []
    plot1_ey = []
    plot1_x = []

    if counts2 != None:
        plot2_y = []
        plot2_ey = []
        plot2_x = []

    # Generate possible configurations
    mymap = {}

    i= 1
    for nf1 in range(0, N+ni+1):
        for nf2 in range(0, N+ni+1-nf1):
            for nphi in range(0, N+ni+1-nf1-nf2):
                mymap[(nf1, nf2, nphi)]= i
                i+= 1

    #print(mymap)
    for c in counts:
        printList= [mycounter, c]
        for n in range(N+ni):
            printList.append(ptype(c.split()[5+n]))
        #print(tuple(printList))
        mycounter+=1

        nf1= 0
        nf2= 0
        nphi= 0

        for n in range(N+ni):
            p= ptype(c.split()[5+n])
            if p == 'f1' or p == 'af1':
                nf1+= 1
            if p == 'f2' or p == 'af2':
                nf2+= 1
            if p == 'phi':
                nphi+= 1


        plot1_y+=[100*counts[c]/events]
        plot1_ey+=[100*counts[c]**0.5/events]
        plot1_x+=[-0.15+mymap[(nf1, nf2, nphi)]]
        

    if counts2 != None:    
        for c in counts2:
            printList= [mycounter, c]
            for n in range(N+ni):
                printList.append(ptype(c.split()[5+n]))
            #print(tuple(printList))
            mycounter+=1

            nf1= 0
            nf2= 0
            nphi= 0

            for n in range(N+ni):
                p= ptype(c.split()[5+n])
                if p == 'f1' or p == 'af1':
                    nf1+= 1
                if p == 'f2' or p == 'af2':
                    nf2+= 1
                if p == 'phi':
                    nphi+= 1


            plot2_y+=[100*counts2[c]/events]
            plot2_ey+=[100*counts2[c]**0.5/events]
            plot2_x+=[0.15+mymap[(nf1, nf2, nphi)]]


    f = plt.figure(figsize=(10, 8))
    ax = f.add_subplot(1, 1, 1)
    plt.ylim((100*1e-4, 100*5.))
    ax.set_yscale("log", nonposy='clip')
    ax.set_ylabel('Probability [%]', fontsize=24)
    bar1 = plt.bar(plot1_x, plot1_y, color='#228b22', width=0.3, label=r"$f' = f_{1}$", hatch='\\') #,yerr=firstisf1_ey)

    plt.xticks(np.arange(1, len(mymap.keys())+1, 1), rotation=45)
    ticklabels= []
    for key in mymap.keys():
        ticklabels+= [str(key)]

    ax.set_xticklabels( tuple(ticklabels) )

    plt.legend(loc='upper right',prop={'size': 9.5})

    if counts2 != None:
        bar2 = plt.bar(plot2_x, plot2_y, color='#FF4949', width=0.3, label=r"$f' = f_{1}$", alpha=1.0) #,hatch="//")
        #plt.bar(firstisf2b_x,firstisf2b_y,color='blue',width=0.4,label=r"$f' = f_{2}$",yerr=firstisf2b_ey,alpha=0.5)
        leg2 = ax.legend([bar1, bar2],[r"$f' = f_{1}, g_{12} = 1$",r"$f' = f_{2}, g_{12} = 1$",r"$f' = f_{1}, g_{12} = 0$"], loc='upper right', prop={'size': 12.5}, frameon=False)
    else:
        leg2 = ax.legend([bar1], [r'$f = f_{a}$',r'$f = f_{b}$'], loc='upper right',frameon=False,prop={'size': 12.5},bbox_to_anchor=(1.,0.8))

    ax.add_artist(leg2);

    plt.title(r"%d-step Full Quantum Simulation" %(N), fontsize=24)
    plt.text(0.1, 0.9, r"$(g_{1},g_{2},\epsilon) = ("+str(g1)+","+str(g2)+",10^{-3})$", fontsize=18, transform=ax.transAxes)

    plt.xlabel(r'$(n_{f_1}, n_{f_2}, n_{\phi})$', fontsize=24)
    #f.savefig("fullsim2step_states.pdf")
    plt.show()