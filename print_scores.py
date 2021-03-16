import matplotlib.pyplot as plt
import numpy as np


def print_scores(scores, agent, episode_step, title):
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22
    # controls default text sizes
    plt.rc('font', family='serif', size=MEDIUM_SIZE)
    plt.rc('font', size=MEDIUM_SIZE)
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    colors = ['b','g','r','c','m']
    des_vel = ['0.6','0.8','1.0','1.2','1.4']

    plt.figure(figsize=(10, 10))
    for i in range(5):
        plt.plot(np.arange(episode_step, len(scores[i])*episode_step+1, episode_step), scores[i], colors[i], linewidth=3, alpha=0.5)
    # plt.scatter(np.arange(1, len(scores)+1), scores,
    #            linewidth=3, c='r', alpha=0.4)
    plt.legend(des_vel, title = "Desired Velocity")
    plt.grid()
    plt.xlabel(r"Number of iterations")
    plt.ylabel(r"Reward")
    plt.title(title)
    ax = plt.gca()
    # ax.set_facecolor('xkcd:salmon')
    #ax.set_facecolor((186.0/255.0, 170.0/255.0, 170.0/255.0))
    # ax.patch.set_facecolor('red')
    # ax.patch.set_alpha(0.4)
    plt.savefig('scores'+str(agent)+'.png')


def print_terrain_scores(scores, agent, episode_step, title):
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22
    # controls default text sizes
    plt.rc('font', family='serif', size=MEDIUM_SIZE)
    plt.rc('font', size=MEDIUM_SIZE)
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    colors = ['b','g','r']
    slopes = ['7','11','15']

    plt.figure(figsize=(10, 10))
    for i in range(3):
        plt.plot(np.arange(episode_step, len(scores[i])*episode_step+1, episode_step), scores[i], colors[i], linewidth=3, alpha=0.5)
    # plt.scatter(np.arange(1, len(scores)+1), scores,
    #            linewidth=3, c='r', alpha=0.4)
    plt.legend(slopes, title = "Percent Grade")
    plt.grid()
    plt.xlabel(r"Number of iterations")
    plt.ylabel(r"Reward")
    plt.title(title)
    ax = plt.gca()
    # ax.set_facecolor('xkcd:salmon')
    #ax.set_facecolor((186.0/255.0, 170.0/255.0, 170.0/255.0))
    # ax.patch.set_facecolor('red')
    # ax.patch.set_alpha(0.4)
    plt.savefig('scores_terr'+str(agent)+'.png')


def print_qvals(scores, agent):
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22
    # controls default text sizes
    plt.rc('font', family='serif', size=MEDIUM_SIZE)
    plt.rc('font', size=MEDIUM_SIZE)
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



    plt.figure(figsize=(10, 10))
    for i in range(5):
        plt.plot(np.arange(1, len(scores)+1, 1), scores,'b', linewidth=3, alpha=0.5)
    # plt.scatter(np.arange(1, len(scores)+1), scores,
    #            linewidth=3, c='r', alpha=0.4)
    plt.grid()
    plt.xlabel(r"Number of iterations")
    plt.ylabel(r"Q-values")
    ax = plt.gca()
    # ax.set_facecolor('xkcd:salmon')
    #ax.set_facecolor((186.0/255.0, 170.0/255.0, 170.0/255.0))
    # ax.patch.set_facecolor('red')
    # ax.patch.set_alpha(0.4)
    plt.savefig('qval'+str(agent)+'.png')

def print_velocities1(test, desired_velocity):
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22
    # controls default text sizes
    plt.rc('font', family='serif', size=MEDIUM_SIZE)
    plt.rc('font', size=MEDIUM_SIZE)
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.figure(figsize=(10, 10))
    plt.plot(np.arange(1, len(test)+1), test, 'b', linewidth=3, alpha=0.5)
    # plt.scatter(np.arange(1, len(scores)+1), scores,
    #            linewidth=3, c='r', alpha=0.4)
    plt.legend([r"Velocity Per Step"])
    plt.grid()
    plt.xlabel(r"Step")
    plt.ylabel(r"Velocity: desired = "+str(desired_velocity))
    ax = plt.gca()
    # ax.set_facecolor('xkcd:salmon')
    #ax.set_facecolor((186.0/255.0, 170.0/255.0, 170.0/255.0))
    # ax.patch.set_facecolor('red')
    # ax.patch.set_alpha(0.4)
    plt.savefig('velocity'+str(desired_velocity)+'.png')

def print_velocities(test, comparison, desired_velocity):
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22
    # controls default text sizes
    plt.rc('font', family='serif', size=MEDIUM_SIZE)
    plt.rc('font', size=MEDIUM_SIZE)
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    vel_arr = [desired_velocity for i in range(len(comparison))]
    plt.figure(figsize=(10, 10))
    plt.plot(np.arange(1, len(test)+1), test, 'b', linewidth=3, alpha=0.5)
    plt.plot(np.arange(1, len(comparison)+1), comparison, 'g', linewidth=3, alpha=0.5)
    plt.plot(np.arange(1, len(vel_arr)+1), vel_arr, 'r', linewidth=7, alpha=0.5)
    # plt.scatter(np.arange(1, len(scores)+1), scores,
    #            linewidth=3, c='r', alpha=0.4)
    plt.legend([r"DDPG",r"On-Policy Control",r"Desired Velocity"])
    plt.grid()
    plt.xlabel(r"Step")
    plt.ylabel(r"Velocity (m/s)")
    ax = plt.gca()
    # ax.set_facecolor('xkcd:salmon')
    #ax.set_facecolor((186.0/255.0, 170.0/255.0, 170.0/255.0))
    # ax.patch.set_facecolor('red')
    # ax.patch.set_alpha(0.4)
    plt.savefig('velocity'+str(desired_velocity)+'.png')


def print_rewards(rew):
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22
    # controls default text sizes
    plt.rc('font', family='serif', size=MEDIUM_SIZE)
    plt.rc('font', size=MEDIUM_SIZE)
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.figure(figsize=(10, 10))
    plt.plot(np.arange(1, len(rew)+1), rew, 'b', linewidth=3, alpha=0.5)
    # plt.scatter(np.arange(1, len(scores)+1), scores,
    #            linewidth=3, c='r', alpha=0.4)
    plt.legend([r"Reward Per Step"])
    plt.grid()
    plt.xlabel(r"Step")
    plt.ylabel(r"Reward")
    ax = plt.gca()
    # ax.set_facecolor('xkcd:salmon')
    #ax.set_facecolor((186.0/255.0, 170.0/255.0, 170.0/255.0))
    # ax.patch.set_facecolor('red')
    # ax.patch.set_alpha(0.4)
    plt.savefig('rewards.png')  
  
def print_theta(scores, episode_step):
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22
    # controls default text sizes
    plt.rc('font', family='serif', size=MEDIUM_SIZE)
    plt.rc('font', size=MEDIUM_SIZE)
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    colors = ['b','g','r','c','m']
    des_vel = ['0.6','0.8','1.0','1.2','1.4']

    plt.figure(figsize=(10, 10))
    for i in range(20):
        plt.plot(np.arange(episode_step, len(scores[i])*episode_step+1, episode_step), scores[i], linewidth=3, alpha=0.5)
    # plt.scatter(np.arange(1, len(scores)+1), scores,
    #            linewidth=3, c='r', alpha=0.4)
    #plt.legend(des_vel)
    plt.grid()
    plt.xlabel(r"Number of iterations")
    plt.ylabel(r"Reward")
    ax = plt.gca()
    # ax.set_facecolor('xkcd:salmon')
    #ax.set_facecolor((186.0/255.0, 170.0/255.0, 170.0/255.0))
    # ax.patch.set_facecolor('red')
    # ax.patch.set_alpha(0.4)
    plt.savefig('theta.png')