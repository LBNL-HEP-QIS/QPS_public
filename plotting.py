import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib
import math

import MakeObservables as mo

def ptype(x):
    '''
    Parses particle type from strings.
    '''
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

def master_plot_emissions(eps, g1, g2, N, ni, events, counts, counts2, mcmc= None, save=True, old_alg=None, wmeas=True):
    '''
    Generates a plot of the number of emissions, like the ones in Figure 6.

    Params:
        g1      (float)             1-type coupling
        g2      (float)             2-type coupling
        N       (int)               number of simulation steps
        ni      (int)               number of initial particles
        events  (int)               number of shots in the simulation
        counts  (dict)              Qiskit simulation results dictionary: with mid-circuit measurements, g12= 0
        counts2 (dict)              Qiskit simulation results dictionary: with mid-circuit measurements, g12= 1
        mcmc    (Array)             Classical MCMC simulation results
        save    (bool)              whether to save the plot as a .pdf
        old_alg (Tuple(dict, dict)) If None: do not overlay Qiskit simulation results WITHOUT mid-circuit measurements.
                                    Otherwise, the tuple consists of two dictionaries containing Qiskit simulation results WITHOUT  mid-circuit measurements:
                                        countsOld for g12= 0, and counts2Old for g12= 1.
        wmeas   (bool)              whether a work register is measured in Qiskit simulations

    Returns:
        None, but displays the plot generated.
    '''
    label = ["24 step simulation ($g_{12} = 0$)", "Simulation ($g_{12} = 0$), Original QPS", "Simulation ($g_{12} = 0$), QPS w/ Remeas.", 
             "IBMQ ($g_{12} = 0$)","24 step simulation ($g_{12} = 1$)", "Simulation ($g_{12} = 1$), Original QPS", "Simulation ($g_{12} = 1$), QPS w/ Remeas.", "IBMQ ($g_{12} = 1$)"]

    f = plt.figure(figsize=(15, 20))

    gs = GridSpec(3, 1, width_ratios=[1], height_ratios=[9, 2, 2])

    ax1 = plt.subplot(gs[0])
    ax1.tick_params(bottom="True", right="False", top="False", left="True", direction='in')


    ######################    
    # simulation         #
    ######################
    emissions_y= np.zeros(N+1)
    emissions_ey= np.zeros(N+1)
    for c in counts:
        emit_list= []
        for n in range(N + ni):
            emit_list= [ptype(c.split()[-1-wmeas-n])] + emit_list

        emit_N= int(mo.Nemissions(emit_list, n_I= ni))

        emissions_y[emit_N]+= counts[c]/events
        emissions_ey[emit_N]+= counts[c]**0.5/events

    emissions_ey= (emissions_y / events)**0.5
    ax1.bar(np.arange(0, N+1, 1), emissions_y, alpha=0.2, color = 'red', width= 1., label=label[6], tick_label= ['']*(N+1))

    emissions2_y= np.zeros(N+1)
    emissions2_ey= np.zeros(N+1)
    for c in counts2:
        emit_list= []
        for n in range(N + ni):
            emit_list= [ptype(c.split()[-1-wmeas-n])] + emit_list

        emit_N= int(mo.Nemissions(emit_list, n_I= ni))

        emissions2_y[emit_N]+= counts2[c]/events
        #emissions2_ey[emit_N]+= counts2[c]**0.5/events # OLD ERROR

    emissions2_ey= (emissions2_y / events)**0.5
    plt.bar(np.arange(0, N+1, 1), emissions2_y, alpha=0.2, color = 'blue', width= 1., label=label[2], tick_label= ['']*(N+1))

    # If also plotting the original algorithm results
    if old_alg != None:
        countsOld= old_alg[0]
        countsOld2= old_alg[1]

        emOld_y= np.zeros(N+1)
        emOld_ey= np.zeros(N+1)
        for c in countsOld:
            emit_list= []
            for n in range(N + ni):
                emit_list= [ptype(c.split()[-1-wmeas-n])] + emit_list

            emit_N= int(mo.Nemissions(emit_list, n_I= ni))

            emOld_y[emit_N]+= countsOld[c]/events
            #emOld_ey[emit_N]+= countsOld[c]**0.5/events # OLD ERROR

        emOld_ey= (emOld_y / events)**0.5
        print('new errors:' + str(emOld_ey))

        ax1.bar(np.arange(0, N+1, 1), emOld_y, alpha=1.0, edgecolor = 'red', color='none', width= 1., linewidth=1.4, 
                label=label[5], tick_label= ['']*(N+1), zorder=10)

        emOld2_y= np.zeros(N+1)
        emOld2_ey= np.zeros(N+1)
        for c in countsOld2:
            emit_list= []
            for n in range(N + ni):
                emit_list= [ptype(c.split()[-1-wmeas-n])] + emit_list

            emit_N= int(mo.Nemissions(emit_list, n_I= ni))

            emOld2_y[emit_N]+= countsOld2[c]/events
            emOld2_ey[emit_N]+= countsOld2[c]**0.5/events

        emOld2_ey= (emOld2_y / events)**0.5
        ax1.bar(np.arange(0, N+1, 1), emOld2_y, alpha=1.0, edgecolor = 'blue', color='none', width= 1., linewidth=1.4, label=label[1], tick_label= ['']*(N+1), zorder=10)

    try:
        plt.scatter(np.arange(0, N+1, 1), mcmc, color= 'black', label=r'Classical MCMC ($g_{12} = 0$)', marker='_', s=1000, linewidths=6, zorder=100)
    except:
        print('Invalid array input for MCMC.')

    # Scale the y-axis. The upper limit is the next closest 0.1 tick.
    if old_alg != None:
        ymax= max(0.65, max(emOld2_y), max(emOld_y), max(emissions_y), max(emissions2_y))
    else:
        ymax= max(0.65, max(emissions_y), max(emissions2_y))
    yceil= math.ceil(10*ymax) / 10. + 0.01
    plt.ylim(0, yceil)
    print('ymax= ' + str(ymax))
    label_format = '{:,.1f}'
    ax1.set_yticks(np.arange(0, yceil, 0.1))
    ax1.set_yticklabels([label_format.format(x) for x in np.arange(0, yceil, 0.1)], size=32, fontname= 'times new roman')

    font = matplotlib.font_manager.FontProperties(family='times new roman', size=25)

    plt.ylabel(r'$\frac{1}{\sigma}$ $\frac{d\sigma}{dE}$', labelpad = 18, fontsize= 72)
    plt.legend(loc='best', prop=font, frameon=False)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)


    ######################
    # ratio plot upper   #
    ######################
    ax2 = plt.subplot(gs[1])
    ax2.set_xticks(np.arange(-1, N+1, 1))
    ax2.tick_params(bottom="True", right="False", top="False", left="True", direction='in')

    if old_alg == None:
        ax2.set_xticklabels(np.arange(-1, N+1, 1), size=32, fontname= 'times new roman')
        plt.xlabel(r'Number of emissions $(E)$', fontsize= 44, labelpad = 18, fontname= 'times new roman')
    else:
        ax2.set_xticklabels(['']*(N+2))

    ax2.step(np.arange(-0.5, N+1.5, 1), [0.]*(N+2), color='black', linestyle='-', linewidth=3)

    diff_g0_2= emissions2_y - mcmc
    ax2.bar(np.arange(0, N+1, 1), diff_g0_2, width= 1., color='blue', alpha=0.2)
    if old_alg != None:
        diff_g0_1= emOld2_y - mcmc
        ax2.step(np.arange(-0.5, N+1.5, 1), np.concatenate((diff_g0_1[:1], diff_g0_1)), color='blue')
        ax2.errorbar(np.arange(0, N+1, 1), np.zeros(N+1), yerr=emOld2_ey, ecolor='blue', elinewidth=3, capsize=6, capthick=2, ls='none')
    else:
        diff_g0_1=np.zeros(N+1)
        ax2.errorbar(np.arange(0, N+1, 1), np.zeros(N+1), yerr=emissions2_ey, ecolor='blue', elinewidth=3, capsize=6, capthick=2, ls='none')

    num_yticks= 3

    max_diff= max(max(abs(diff_g0_1)), max(abs(diff_g0_2)))
    dy= math.ceil(1000 * max_diff) / 1000

    ytick_range= np.arange(-dy*(math.floor(num_yticks/2.)), dy*(math.floor(num_yticks/2.)+0.5), dy)
    ax2.set_yticks(ytick_range)
    label_format = '{:,.3f}'
    ax2.set_yticklabels([label_format.format(x) for x in ytick_range], size=28, fontname= 'times new roman')
    plt.ylabel('Quantum  -\n Classical', fontsize= 35, labelpad = 18, fontname= 'times new roman')

    if old_alg != None:
        ax3 = plt.subplot(gs[2])
        ax3.tick_params(bottom="True", right="False", top="False", left="True", direction='in')

        label_format = '{:,.1f}'
        ax3.set_xticks(np.arange(-1, N+1, 1))
        ax3.set_xticklabels(np.arange(-1, N+1, 1), size=32, fontname= 'times new roman')
        plt.xlabel(r'Number of emissions $(E)$', fontsize= 44, labelpad = 18, fontname= 'times new roman')

        ax3.step(np.arange(-0.5, N+1.5, 1), [0.]*(N+2), color='red', linestyle='-', linewidth= 3, zorder=1)
        ax3.step(np.arange(-0.5, N+1.5, 1), [0.]*(N+2), color='blue', linestyle=':', linewidth= 3, zorder=2)

        diff_old1= emissions_y - emOld_y
        diff_old2= emissions2_y - emOld2_y
        ax3.bar(np.arange(0, N+1, 1), diff_old1, width= 1., color='red', alpha=0.2)
        ax3.bar(np.arange(0, N+1, 1), diff_old2, width= 1., color='blue', alpha=0.2)

        # Scale y-axis.
        max_diff= max(max(abs(diff_old1)), max(abs(diff_old2)))
        dy= math.ceil(1000 * max_diff) / 1000
        ytick_range= np.arange(-dy*(math.floor(num_yticks/2.)), dy*(math.floor(num_yticks/2.)+0.1), dy)
        ax3.set_yticks(ytick_range)
        label_format = '{:,.3f}'
        ax3.set_yticklabels([label_format.format(x) for x in ytick_range], size=28, fontname= 'times new roman')

        # Error of difference (variances add)
        ax3.errorbar(np.arange(0.1, N+1.1, 1), np.zeros(N+1), yerr=(emOld_ey**2 + emissions_ey**2)**0.5, ecolor='red', elinewidth=3, capsize=6, capthick=2, ls='none')
        ax3.errorbar(np.arange(-0.1, N+0.9, 1), np.zeros(N+1), yerr=(emOld2_ey**2 + emissions2_ey**2)**0.5, ecolor='blue', elinewidth=3, capsize=6, capthick=2, ls='none')

        plt.ylabel('Remeas. - \n Original', fontsize= 35, labelpad = 18, fontname= 'times new roman')
        ax3.set_yticks(ytick_range)

    gs.update(wspace=0.025, hspace=0.2)

    if save == True:
        if old_alg != None:
            plt.savefig('./data/sim%dstep_emissions_shots=%s_paperStyle.pdf' %(N, '{:.0e}'.format(events)), bbox_inches='tight')
        else:
            plt.savefig('./data/sim%dstep_emissions_shots=%s_paperStyle_NEW.pdf' %(N, '{:.0e}'.format(events)), bbox_inches='tight')
    plt.show()





def master_plot_thetamax(eps, g1, g2, N, ni, events, counts, counts2, analytical= None, save=True, normalized=False, old_alg=None, wmeas=True):
    '''
    Generates a plot of the hardest emissions θmax, like the ones in Figure 7.

    Params:
        eps        (float)             Discretization parameter, see Section IV Eq. (17).
        g1         (float)             1-type coupling
        g2         (float)             2-type coupling
        N          (int)               number of simulation steps
        ni         (int)               number of initial particles
        events     (int)               number of shots in the simulation
        counts     (dict)              Qiskit simulation results dictionary: with mid-circuit measurements, g12= 0
        counts2    (dict)              Qiskit simulation results dictionary: with mid-circuit measurements, g12= 1
        analytical (Array)             Analytical hardest emission curve (computed classically)
        save       (bool)              whether to save the plot as a .pdf
        normalized (bool)              whether to normalize conditionally based on at least one emission occurring.
        old_alg    (Tuple(dict, dict)) If None: do not overlay Qiskit simulation results WITHOUT mid-circuit measurements.
                                       Otherwise, the tuple consists of two dictionaries containing Qiskit simulation results WITHOUT  mid-circuit measurements:
                                           countsOld for g12= 0, and counts2Old for g12= 1.
        wmeas   (bool)              whether a work register is measured in Qiskit simulations

    Returns:
        None, but displays the plot generated.
    '''

    label = ["24 step simulation ($g_{12} = 0$)", "Simulation ($g_{12} = 0$), Original QPS", "Simulation ($g_{12} = 0$), QPS w/ Remeas.", 
             "IBMQ ($g_{12} = 0$)","24 step simulation ($g_{12} = 1$)", "Simulation ($g_{12} = 1$), Original QPS", "Simulation ($g_{12} = 1$), QPS w/ Remeas.", "IBMQ ($g_{12} = 1$)"]


    f = plt.figure(figsize=(15, 18))

    gs = GridSpec(3, 1, width_ratios=[1], height_ratios=[9, 2, 2])
    ax1 = plt.subplot(gs[0])
    ax1.tick_params(bottom="True", right="False", top="False", left="True", direction='in')

    ######################    
    # simulation         #
    ######################
    hist_bins, centers= mo.hist_bins(N, eps)

    dx= hist_bins[-1] - hist_bins[-2]
    x_axis= np.arange(math.floor(hist_bins[0]), math.ceil(hist_bins[-1]) + 1, 1)

    # Center ticks
    ax1.set_xticks(centers)
    ax1.set_xticklabels(['{:,.2f}'.format(x) for x in centers], size=28, fontname= 'times new roman')

    tm_x= centers
    tm_ey= np.zeros(N)
    tm_y= np.zeros(N)

    for c in counts:
        emit_list= []
        for n in range(0, N + ni):
            emit_list= [ptype(c.split()[-1-wmeas-n])] + emit_list
        theta_max, _, _= mo.LogThetaMax(emit_list, n_I= ni, eps= eps)
        if theta_max != None:
            j= 0
            while hist_bins[j+1] < theta_max:
                j+= 1

            tm_y[j]+= counts[c]/events
            #tm_ey[j]+= counts[c]**0.5/events # OLD ERROR
    tm_ey= (tm_y / events)**0.5
    if normalized:
        ax1.bar(tm_x, tm_y/dx, alpha=0.2, color='red', width= dx, linewidth=1.4, label=label[6], zorder=1)
    else:
        ax1.bar(tm_x, tm_y, alpha=0.2, color='red', width= dx, linewidth=1.4, label=label[6], tick_label= ['']*N, zorder=1)
    
    if counts2 != None:
        tm2_x= centers
        tm2_ey= np.zeros(N)
        tm2_y= np.zeros(N)
    
        for c in counts2:
            emit_list= []
            for n in range(0, N + ni):
                emit_list= [ptype(c.split()[-1-wmeas-n])] + emit_list

            theta_max, _, _= mo.LogThetaMax(emit_list, n_I= ni, eps= eps)
            if theta_max != None:
                j= 0
                while hist_bins[j+1] < theta_max:
                    j+= 1

                tm2_y[j]+= counts2[c]/events
                #tm2_ey[j]+= counts2[c]**0.5/events # OLD ERROR
        tm2_ey= (tm2_y / events)**0.5
        if normalized:
            ax1.bar(tm_x, tm2_y/dx, alpha=0.2, color='blue', width= dx, linewidth=1.4, label=label[2], zorder=1)
        else:
            ax1.bar(tm_x, tm2_y, alpha=0.2, color='blue', width= dx, linewidth=1.4, label=label[2], tick_label= ['']*N, zorder=1)

    if old_alg != None:
        countsOld= old_alg[0]
        countsOld2= old_alg[1]
        tmOld_y= np.zeros(N)
        tmOld_ey= np.zeros(N)
        tmOld2_y= np.zeros(N)
        tmOld2_ey= np.zeros(N)

        for c in countsOld:
            emit_list= []
            for n in range(0, N + ni):
                emit_list= [ptype(c.split()[-1-wmeas-n])] + emit_list
            theta_max, _, _= mo.LogThetaMax(emit_list, n_I= ni, eps= eps)
            if theta_max != None:
                j= 0
                while hist_bins[j+1] < theta_max:
                    j+= 1

                tmOld_y[j]+= countsOld[c]/events
                # tmOld_ey[j]+= countsOld[c]**0.5/events # OLD ERROR
        tmOld_ey= (tmOld_y / events)**0.5
        if normalized:
            ax1.bar(tm_x, tmOld_y/dx, alpha=1.0, edgecolor = 'red', color='none', width= dx, linewidth=1.4, label=label[5], zorder=10)
        else:
            ax1.bar(tm_x, tmOld_y, alpha=1.0, edgecolor = 'red', color='none', width= dx, linewidth=1.4, label=label[5], zorder=10)

        for c in countsOld2:
            emit_list= []
            for n in range(0, N + ni):
                emit_list= [ptype(c.split()[-1-wmeas-n])] + emit_list

            theta_max, _, _= mo.LogThetaMax(emit_list, n_I= ni, eps= eps)
            if theta_max != None:
                j= 0
                while hist_bins[j+1] < theta_max:
                    j+= 1

                tmOld2_y[j]+= countsOld2[c]/events
                #tmOld2_ey[j]+= countsOld2[c]**0.5/events # OLD ERROR
        tmOld2_ey= (tmOld2_y / events)**0.5
        if normalized:
            ax1.bar(tm_x, tmOld2_y/dx, alpha=1.0, edgecolor = 'blue', color='none', width= dx, linewidth=1.4, label=label[1], zorder=10)
        else:
            ax1.bar(tm_x, tmOld2_y, alpha=1.0, edgecolor = 'blue', color='none', width= dx, linewidth=1.4, label=label[1], tick_label= ['']*N, zorder=10)

    try:
        plt.plot(analytical[0], analytical[1], color= 'black', label=r'Analytical ($g_{12} = 0$)', linestyle='--', linewidth=2., zorder=100)
    except:
        print('Invalid array input for MCMC.')

    # Scale the y-axis. The upper limit is the next closest 0.1 tick.
    if old_alg != None:
        ymax= max(0.5, max(tmOld2_y), max(tmOld_y), max(tm_y), max(tm2_y))
    else:
        ymax= max(0.5, max(tm_y), max(tm2_y))
    if normalized:
        ymax= max(0.5, ymax/dx)

    yceil= math.ceil(10*ymax) / 10. + 0.01
    plt.ylim(0, yceil)
    print('ymax= ' + str(ymax))
    label_format = '{:,.1f}'
    ax1.set_yticks(np.arange(0, yceil, 0.1))
    ax1.set_yticklabels([label_format.format(x) for x in np.arange(0, yceil, 0.1)], size=32, fontname= 'times new roman')

    font = matplotlib.font_manager.FontProperties(family='times new roman', size=30)

    plt.ylabel(r'$\frac{1}{\sigma}$ $\frac{d\sigma}{d\log(\theta_{\max})}$', labelpad = 18, fontsize= 60)
    plt.legend(loc='best', prop=font, frameon=False)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)

    ######################
    # ratio plot upper   #
    ######################
    ax2 = plt.subplot(gs[1])
    ax2.tick_params(bottom="True", right="False", top="False", left="True", direction='in')
    ax2.set_xticks(x_axis)
    if old_alg == None:
        ax2.set_xticklabels(x_axis, size=28, fontname= 'times new roman')
        label_format = '{:,.1f}'
        ax2.set_xlabel(r'$\log_e(\theta_{\mathrm{max}})$', fontsize= 40, labelpad = 19, fontname= 'times new roman')
    else:
        ax2.set_xticklabels(['']*len(x_axis))

    ax2.step(np.arange(centers[0] - dx/2., centers[-1] + dx/2. + 1e-14, dx), [0.]*(N+1), color='black', linestyle='-', linewidth=3)

    mcmc= np.zeros(N)
    Nind= np.size(analytical[1])
    for j in range(N):
        start= np.where(hist_bins[-j-1] > analytical[0])[0][0]
        endArray= np.where(hist_bins[-j-2] > analytical[0])[0]
        if np.size(endArray) == 0:
            mcmc[N-j-1]= np.sum(analytical[1][start:])
        else:
            print(start, endArray[0])
            mcmc[N-j-1]= np.sum(analytical[1][start:endArray[0]])

    mcmc/= np.size(analytical[1]) / N
    diff_g0_2= tm2_y / dx - mcmc

    ax2.bar(centers, diff_g0_2, width= dx, color='blue', alpha=0.2)
    if old_alg != None:
        diff_g0_1= tmOld2_y / dx - mcmc
        if normalized:
            ax2.step(np.arange(centers[0] - dx/2., centers[-1] + dx/2. + 1e-14, dx), np.concatenate((diff_g0_1[:1], diff_g0_1)), color='blue')
            ax2.errorbar(centers, np.zeros(N), yerr=tmOld2_ey/dx, ecolor='blue', elinewidth=3, capsize=6, capthick=2, ls='none')
        else:
            ax2.step(np.arange(centers[0] - dx/2., centers[-1] + dx/2. + 1e-14, dx), np.concatenate((diff_g0_1[:1], diff_g0_1))*dx, color='blue')
            ax2.errorbar(centers, np.zeros(N), yerr=tmOld2_ey, ecolor='blue', elinewidth=3, capsize=6, capthick=2, ls='none')
    else:
        diff_g0_1=np.zeros(N+1)
        if normalized:
            ax2.errorbar(centers, np.zeros(N), yerr=tm2_ey/dx, ecolor='blue', elinewidth=3, capsize=6, capthick=2, ls='none')
        else:
            ax2.errorbar(centers, np.zeros(N), yerr=tm2_ey, ecolor='blue', elinewidth=3, capsize=6, capthick=2, ls='none')

    num_yticks= 3

    # Scale the y-axis.
    max_diff= max(max(abs(diff_g0_1)), max(abs(diff_g0_2)))
    dy= math.ceil(1000 * max_diff) / 1000
    ytick_range= np.arange(-dy*(math.floor(num_yticks/2.)), dy*(math.floor(num_yticks/2.)+0.5), dy)
    ax2.set_yticks(ytick_range)
    label_format = '{:,.3f}'
    ax2.set_yticklabels([label_format.format(x) for x in ytick_range], size=28, fontname= 'times new roman')
    plt.ylabel('Quantum  -\n Classical', fontsize= 35, labelpad = 18, fontname= 'times new roman')

    ######################
    # ratio plot lower   #
    ######################
    if old_alg != None:
        ax3 = plt.subplot(gs[2])
        ax3.tick_params(bottom="True", right="False", top="False", left="True", direction='in')

        ax3.set_xticks(x_axis)
        ax3.set_xticklabels(x_axis, size=28, fontname= 'times new roman')

        ax3.set_yticks(np.arange(0, 3.5, 2))
        ax3.set_yticklabels([label_format.format(x) for x in np.arange(0, 3.5, 2)], size=28, fontname= 'times new roman')

        ratio1= tm_y / tmOld_y
        ratio2= tm2_y / tmOld2_y

        diff1= tm_y - tmOld_y
        diff2= tm2_y - tmOld2_y
        
        # Plot differences
        if normalized:
            ax3.bar(centers, diff1 / dx, width= dx, color='red', alpha=0.2)
            ax3.bar(centers, diff2 / dx, width= dx, color='blue', alpha=0.2)
        else:
            ax3.bar(centers, diff1, width= dx, color='red', alpha=0.2)
            ax3.bar(centers, diff2, width= dx, color='blue', alpha=0.2)

        ax3.step(np.arange(centers[0] - dx/2., centers[-1] + dx/2. + 1e-14, dx), [0.]*(N+1), color='red', linestyle='-', linewidth= 3, zorder=1)
        ax3.step(np.arange(centers[0] - dx/2., centers[-1] + dx/2. + 1e-14, dx), [0.]*(N+1), color='blue', linestyle=':', linewidth= 3, zorder=2)

        if normalized:
            ax3.errorbar(centers+0.1, np.zeros(N), yerr=(tmOld_ey**2 + tm_ey**2)**0.5 / dx, ecolor='red', elinewidth=3, capsize=6, capthick=2, ls='none')
            ax3.errorbar(centers-0.1, np.zeros(N), yerr=(tmOld2_ey**2 + tm2_ey**2)**0.5 / dx, ecolor='blue', elinewidth=3, capsize=6, capthick=2, ls='none')
        else:
            ax3.errorbar(centers+0.1, np.zeros(N), yerr=(tmOld_ey**2 + tm_ey**2)**0.5, ecolor='red', elinewidth=3, capsize=6, capthick=2, ls='none')
            ax3.errorbar(centers-0.1, np.zeros(N), yerr=(tmOld2_ey**2 + tm2_ey**2)**0.5, ecolor='blue', elinewidth=3, capsize=6, capthick=2, ls='none')

        # Scale the y-axis.
        max_diff= max(max(abs(diff1)), max(abs(diff2))) / dx
        dy= math.ceil(1000 * max_diff) / 1000 
        ytick_range= np.arange(-dy*(math.floor(num_yticks/2.)), dy*(math.floor(num_yticks/2.)+0.5), dy)
        ax3.set_yticks(ytick_range)
        label_format = '{:,.3f}'
        ax3.set_yticklabels([label_format.format(x) for x in ytick_range], size=28, fontname= 'times new roman')

        plt.ylabel('Remeas. - \n Original', fontsize= 35, labelpad = 18, fontname= 'times new roman')
        ax3.set_xlabel(r'$\log_e(\theta_{\mathrm{max}})$', fontsize= 40, labelpad = 19, fontname= 'times new roman')
       
    gs.update(wspace=0.025, hspace=0.2)

    if save == True:
        if old_alg != None:
            plt.savefig('./data/sim%dstep_thetamax_shots=%s_paperStyle.pdf' %(N, '{:.0e}'.format(events)), bbox_inches='tight')
        else:
            plt.savefig('./data/sim%dstep_thetamax_shots=%s_paperStyle_NEW.pdf' %(N, '{:.0e}'.format(events)), bbox_inches='tight')
    plt.show()