import numpy as np
import scipy.io as sio

def theta_transform(theta):
        upplim_jthigh = 250*(np.pi/180)
        lowlim_jthigh = 90*(np.pi/180)
        upplim_jleg = 120*(np.pi/180)
        lowlim_jleg = 0

        thigh_const = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        leg_const = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        low_lim = (thigh_const*lowlim_jthigh) + (lowlim_jleg*leg_const)
        high_lim = (thigh_const*upplim_jthigh) + (upplim_jleg*leg_const)
        diff_lim = high_lim - low_lim

        #subtract lower limit
        theta = theta - low_lim
        #Divide by difference
        theta = theta / diff_lim

        return theta

correct_theta_6 = np.array([ 0.64155571, -1.81932965,  1.21838243, -2.72523574, -0.90401907, -0.54914252, -1.3982528,   0.14749336, -2.26035832, -0.39200816,  1.61652063,  0.07386109, 2.40301629, -2.15418479, -3.646863,    0.41908823, -1.81067398, -1.83107217, -2.0306081,  -3.4525185 ])

correct_theta_7 = np.array( [ 0.51973223, -1.71943865,  1.131848,   -2.6750585,  -0.89112097, -0.47822295, -1.48189, 0.09039083, -2.26783792, -0.39007847,  1.53535213,  0.0419156,2.22780702, -2.03242455, -3.54660611,  0.37162875, -1.76806187, -1.79998285,-2.0295311,  -3.32484209])

correct_theta_8 = np.array([ 0.26185508, -1.49490969,  0.93030186, -2.57938559, -0.90701633, -0.33876894,-1.71705503, -0.04654146, -2.31240617, -0.38186279,  1.32392515, -0.05646301,1.8128406,  -1.73397957, -3.35630595,  0.29082569, -1.6766745 , -1.75126384,-2.05198083, -3.0386785 ])

correct_theta_9 = np.array([ 0.09125547, -1.32454292,  0.76630949, -2.53368899, -0.98963103, -0.26431992,-1.96963501, -0.1639293,  -2.38985065, -0.36953309,  1.1179467,  -0.17287855,  1.46474424, -1.46860979, -3.26700498,  0.27018265, -1.61423801, -1.74755923,-2.10810686, -2.82283344])

correct_theta_10 = np.array([ 0.0778334,  -1.31635,   0.76072591, -2.52589332, -0.97891978, -0.25421173,-1.96635038, -0.16676742, -2.38449491, -0.37020888,  1.11752209, -0.1697838,1.45492052, -1.46394799, -3.25124394,  0.26072574, -1.60979814, -1.74045831,-2.10266987, -2.81218531])

correct_theta_11 = np.array([-0.00395217, -1.22180378,  0.66402862, -2.51436267, -1.06104159, -0.22902164,-2.14464091, -0.23884685, -2.44990361, -0.36023233,  0.97979241, -0.25586407,1.244657,   -1.29667032, -3.2300115,   0.27017876 ,-1.57869914, -1.75550368,-2.15391589, -2.69306304])

correct_theta_12 = np.array([-0.14817836, -1.10749811,  0.56713545, -2.45177017, -1.0327093,  -0.14183415,-2.22608389, -0.30159489, -2.45006965, -0.35919688,  0.89567436, -0.28438413,1.05055713, -1.16482642, -3.10468831,  0.20804703, -1.52860938, -1.71352916,-2.1451631,  -2.54671454])

correct_theta_13 = np.array([-0.24991701, -1.03072263,  0.50420311, -2.40450771, -0.99998421, -0.07718481,-2.26639465, -0.34112275, -2.44171267, -0.35968467,  0.84801728, -0.29543246,0.92663684, -1.08382535, -3.00981844,  0.158421,   -1.49362519, -1.6788797,-2.13169602, -2.44816681])

correct_theta_14 = np.array([-0.34613193, -0.96447641,  0.45362146, -2.3546838,  -0.94802637, -0.0108565,-2.27624956, -0.37069511, -2.41983394, -0.36215504,  0.82221143, -0.29092267,0.83088384, -1.02701746, -2.9094363,   0.10192784, -1.46111703, -1.63779916,-2.10693286, -2.36270107])

correct_theta_15 = np.array([-0.40809593, -0.91989413,  0.41835134, -2.32414296, -0.92090169,  0.03029489,-2.29112246, -0.39209573, -2.40995905, -0.36314002,  0.79978156, -0.29253012,0.76275065, -0.98446141, -2.84800481,  0.06842932 ,-1.44000732, -1.61384983,-2.09461272, -2.30532768])




def choseOfflineAction(desired_velocity):
        if desired_velocity < .65:
                correct_theta = correct_theta_6
        elif desired_velocity < .75:
                correct_theta = correct_theta_7
        elif desired_velocity < .85:
                correct_theta = correct_theta_8
        elif desired_velocity < .95:
                correct_theta = correct_theta_9
        elif desired_velocity < 1.05:
                correct_theta = correct_theta_10
        elif desired_velocity < 1.15:
                correct_theta = correct_theta_11
        elif desired_velocity < 1.25:
                correct_theta = correct_theta_12
        elif desired_velocity < 1.35:
                correct_theta = correct_theta_13
        elif desired_velocity < 1.45:
                correct_theta = correct_theta_14
        else:
                correct_theta = correct_theta_15
        return correct_theta

def sigmoid(x):
        return 1 / (1 + np.exp(-x))