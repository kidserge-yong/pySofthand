import csv
import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt 
import math

inv = lambda n: np.linalg.inv(n)
tp = lambda n: np.transpose(n)
dot = lambda x, y: np.dot(x,y)

DEBUG = False

class synergy:

    init_method = 'random'
    solver = 'mu'
    max_iter = 1000
    H = np.array([])

    
    def __init__(self):
        print("Synergy: %s has been built." % __name__)
        pass

    def fit(self, Xs, ys):
        """
        Fit synergy model.

        Parameters
        ----------
        Xs : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data

        ys : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary

        Returns
        -------
        self : returns an instance of self.
        """
        assert len(Xs) != 0, "Xs is empty"
        assert len(ys) != 0, "ys is empty"
        assert len(Xs) == len(ys), "Xs and ys have different size"

        flat_x = np.hstack([X for X in Xs])
        #flat_y = np.hstack([y for y in ys])
        nflat_x, self.normal_X_range = self.__normalize(flat_x)
        #plot(tp(nflat_x))
        index = 1

        for X, y in zip(Xs, ys):
            normal_X, _ = self.__normalize(X, self.normal_X_range)
            #normal_y, _ = self.__normalize(y, None)

            opti_com = self.__findoptimalcomp(normal_X, y)
            #opti_com = 3
            model = NMF(n_components = opti_com, init = self.init_method, solver = self.solver, max_iter = self.max_iter)
            
            model.fit_transform(tp(normal_X))
            H, _ = self.__normalize(model.components_)
            H = np.array(tp(H))
            #print(len(H), len(H[0]))
            #print(len(tem), len(tem[0]))
            #multiplot(tem)

            #self.H = np.hstack([self.H, H]) if self.H.size else H
            self.H = np.concatenate((self.H, H), axis=1) if self.H.size else H

            if DEBUG:
                print("normal_X size = %d, %d" % (len(normal_X), len(normal_X[0])))
                print("Synergy %d have %d component" % (index, opti_com))
                index += 1 
                multiplot(self.__synergyComputation(nflat_x, H))
                #print(len(self.H))
                #print(self.H, model.components_)
                pass

        
        synergy = self.__synergyComputation(nflat_x, self.H)
        normal_synergy, self.normal_synergy_range = self.__normalize(tp(synergy))
        
        if DEBUG:
            print("nflat_x size = %d, %d" % (len(nflat_x), len(nflat_x[0])))
            print("self.H size = %d, %d" % (len(self.H), len(self.H[0])))
            multiplot(tp(normal_synergy))

        return self

    def transform(self, source):
        assert len(source) != 0, "source is empty"

        nsource, _ = self.__normalize(source, self.normal_X_range)
        synergy = self.__synergyComputation(nsource, self.H)
        synergy, self.normal_synergy_range = self.__normalize(tp(synergy), self.normal_synergy_range)
        #synergy, _ = self.__normalize(tp(ans), self.normal_synergy_range)
        #return tp(synergy)
        return synergy

    def __synergyComputation(self, source, synH):
        """
        Calculate synergy and have option for increase performance.

        Parameters
        ----------
        source : {array-like, sparse matrix} of shape (channel, time)
            data source to transform to synergy

        synH : array-like of shape (channel, component)
            transform matrix according to NMF model

        Returns
        -------
        synergy : returns transformed synergy.
        """
        source = tp(source)
        e = 0.001
        #print("source = %d, %d" % (len(source), len(source[0])))
        #print("synH = %d, %d" % (len(synH), len(synH[0])))
        invStS = inv(np.dot(tp(synH), synH))
        #print("invStS = %d, %d" % (len(invStS), len(invStS[0])))
        invStSSt = np.dot(invStS, tp(synH))
        #print("invStSSt = %d, %d" % (len(invStSSt), len(invStSSt[0])))
        synergy = []
        for i in range(len(source)):
            eps = e
            Tx = tp(np.dot(invStSSt, tp(source[i])))
            count = 0   
            # while((Tx < -eps).any()):
            #     minus = np.where((Tx < -eps)==True)
            #     plus = np.where((Tx < -eps)==False)
            #     minus = minus[0]
            #     plus = plus[0]

            #     # delete channel that is minus
            #     new_S = np.delete(synH, minus, 1)
            #     cut = np.array([0.0]*len(Tx))

            #     num = np.dot(synH[:, minus], tp(abs(Tx[minus])))
            #     num = np.dot(tp(new_S), num)
            #     den = inv(np.dot(tp(new_S),new_S))
            #     test = np.dot(den, num)

            #     for j, item in enumerate(plus):
            #         cut[item] = test[j]
            #         #print(i, item)

            #     # for j, item in enumerate(minus):
            #     #     Tx[item] = 0.0
            #         #print(i, item)

            #     Tx = Tx - cut
            #     count += 1
            #     eps = e*10**(np.floor(math.log10(count)))
            # #     #print(Tx)
            synergy.append(Tx)
        #print("Tx = %d" % (len(Tx)))
        return synergy
        #return tp(np.dot(invStSSt, tp(source)))


    def __findoptimalcomp(self, X, y, max_components = 10, cc_criteria = 0.9, tol=0.00001):
        
        n_components = 2
        max_cc = 0
        max_cc_component = 0

        while n_components < max_components:
            n_components = n_components + 1
            model = NMF(n_components = n_components, init = self.init_method, solver = self.solver, max_iter = self.max_iter)
            W = model.fit_transform(tp(X))
            H = model.components_

            predict = np.dot(H, X)

            cc = self.__efficientindicator(predict, y)
            if cc > cc_criteria:
                return n_components
            if cc > max_cc:
                max_cc = cc
                max_cc_component = n_components
            
            print("cc = %f, n_components = %d" % (cc, n_components))

            if DEBUG:
                pass
                
        return max_cc_component

    def __efficientindicator(self, X, y):
        import scipy.stats
        cc = []
        for y_channel in y:
            y_pos = [i if i > 0 else 0 for i in y_channel]
            y_neg = [-i if i < 0 else 0 for i in y_channel]
            accu_cc = []
            for x_channel in X:
                result_pos = scipy.stats.linregress(x_channel, y_pos)
                result_neg = scipy.stats.linregress(x_channel, y_neg)

                accu_cc.append((abs(result_pos.rvalue) + abs(result_neg.rvalue)) / 2)
                if DEBUG:
                    #print("x_channel, y_pos")
                    #plot(tp([x_channel, y_pos]))
                    #print("x_channel, y_neg")
                    #plot(tp([x_channel, y_neg]))
                    pass
            cc.append(sum(accu_cc)/len(accu_cc))
            if DEBUG:
                #plot(x_channel)
                #plot(y_pos)
                #plot(y_neg)
                #print(cc)
                pass
           
        return max(cc)

    def __normalize(self, data, channel_range = None):
        """

        """
        
        ans = []
        if channel_range == None:
            channel_range = [(min(channel), max(channel)) for channel in data]
        for ich in range(len(data)):
            range_value = float(channel_range[ich][1]-channel_range[ich][0])
            if range_value == 0:
                normal_data = data[ich] - data[ich]
            else:
                normal_data = (data[ich] - channel_range[ich][0]) / range_value
            for inor in range(len(normal_data)):
                if normal_data[inor] < 0:
                    normal_data[inor] = 0.0
            ans.append(normal_data)
        return ans, channel_range

def plot(graph):
    plt.plot(graph)
    plt.show() 

def multiplot(graph, channel_dir = 1):
    if channel_dir > 0:
        graph = tp(graph)
    fig, axs = plt.subplots(len(graph), 1)
    for i, item in enumerate(graph):
        axs[i].plot(item)
    plt.show()

def array_normalize(data, array_min=None, array_max=None):
    """
    data[[time] x channel]: data of each channel and time
    """
    ans = []
    if not array_max == None:
        max_ans = array_max
    else:
        max_ans = [max(channel) for channel in data]
    if not array_min == None:
        min_ans = array_min
    else:
        min_ans = [min(channel) for channel in data]
    data = np.array(data)       # cast to numpy array
    for ich in range(len(data)):
        range_value = float(max_ans[ich]-min_ans[ich])
        if range_value == 0:
            normal_data = data[ich] - data[ich]
        else:
            normal_data = (data[ich] - min_ans[ich]) / range_value
        for inor in range(len(normal_data)):
            if normal_data[inor] < 0:
                normal_data[inor] = 0.0
        ans.append(normal_data)
    return ans, min_ans, max_ans




if __name__ == "__main__":
    import os

    filename = os. getcwd() + '/../data/train_data_09_17_2020_11_13_54.csv'
    data = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                data.append([float(value) for value in row])
            except:
                #print(row)
                pass

    EMGdata = [item[:-6] for item in data]
    Motiondata = [item[-6:] for item in data]

    data_X = []
    data_Y = []
    tem_X = []
    tem_Y = []
    pre_j = Motiondata[0]

    for i,j in zip(EMGdata,Motiondata):
        #print(j)
        if j != pre_j:
            data_X.append(tp(tem_X))
            data_Y.append(tp(tem_Y))
            tem_X = []
            tem_Y = []
        tem_X.append(i)
        tem_Y.append(j)
        pre_j = j

    data_X.append(tp(tem_X))
    data_Y.append(tp(tem_Y))

    nEMGdata, minEMGdata, maxEMGdata = array_normalize(tp(EMGdata))

    # for i in range(len(data_X)):
    #     data_X[i] = np.array(data_X[i])
    # for i in range(len(data_Y)):
    #     data_Y[i] = np.array(data_Y[i])

    gripdata_X = np.concatenate((data_X[0], data_X[1], data_X[2]), axis=1)
    gripdata_X, tem2 , tem3 = array_normalize(gripdata_X, minEMGdata, maxEMGdata)
    gripdata_Y = np.concatenate((data_Y[0], data_Y[1], data_Y[2]), axis=1)

    wristdata_X = np.concatenate((data_X[0], data_X[3], data_X[4]), axis=1)
    wristdata_X, tem2 , tem3 = array_normalize(wristdata_X, minEMGdata, maxEMGdata)
    wristdata_Y = np.concatenate((data_Y[0], data_Y[3], data_Y[4]), axis=1)

    prosudata_X = np.concatenate((data_X[0], data_X[5], data_X[6]), axis=1)
    prosudata_X, tem2 , tem3 = array_normalize(prosudata_X, minEMGdata, maxEMGdata)
    prosudata_Y = np.concatenate((data_Y[0], data_Y[5], data_Y[6]), axis=1)

    sum_x = [gripdata_X, wristdata_X, prosudata_X]
    sum_y = [gripdata_Y, wristdata_Y, prosudata_Y]

    synergy = synergy()
    synergy.fit(sum_x, sum_y)
    transform_y = synergy.transform(nEMGdata)
    #plot(EMGdata)
    #plot(transform_y)

    ny, _, _ = array_normalize(transform_y)

    multiplot(tp(ny))
    #multiplot(tp(tp(ny)[1:10])) 

    transform_y = synergy.transform(gripdata_X)


    

