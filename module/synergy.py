import csv
import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt 
import math

inv = lambda n: np.linalg.inv(n)
tp = lambda n: np.transpose(n)
dot = lambda x, y: np.dot(x,y)

DEBUG = False

class synergy():

    init_method = 'random'
    solver = 'mu'
    max_iter = 10000
    tol = 5e-3
    H = np.array([])

    
    def __init__(self):
        print("Synergy: %s has been built." % __name__)
        pass

    def fit(self, Xs, ys):
        """
        Fit synergy model.

        Parameters
        ----------
        Xs : {list of array-like, sparse matrix} of shape (n_samples, n_features)
            Training data

        ys : list of array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary

        Returns
        -------
        self : returns an instance of self.
        """
        assert len(Xs) != 0, "Xs is empty"
        assert len(ys) != 0, "ys is empty"
        assert len(Xs) == len(ys), "Xs and ys have different size"

        flat_x = np.vstack([X for X in Xs])
        #flat_y = np.hstack([y for y in ys])
        nflat_x, self.normal_X_range = self.__normalize(flat_x)
        index = 1

        for X, y in zip(Xs, ys):
            #normal_X, _ = self.__normalize(X, self.normal_X_range)
            #normal_y, _ = self.__normalize(y, None)

            opti_com = self.__findoptimalcomp(X, y)
            #opti_com = 3

            # model = NMF(n_components = opti_com, init = self.init_method, solver = self.solver, max_iter = self.max_iter, tol = self.tol)
            # model.fit_transform(tp(normal_X))
            # H, _ = self.__normalize(model.components_)
            # H = np.array(tp(H))

            #W, H, RR = self.__NMF(tp(normal_X), opti_com)
            #print(len(H), len(H[0]))
            #print(len(tem), len(tem[0]))

            X = np.array(X)

            channel_num = X.shape[1]
            sample_num = X.shape[0]
            component_num = opti_com
            

            from numpy import random
            U = abs(random.random([sample_num, component_num]))
            V = abs(random.random([channel_num, component_num]))

            self.U, self.V, self.CPUtime, self.NRV, self.RRV = self.fasthals(X, U, V, 1000)
            H = V

            #self.H = np.hstack([self.H, H]) if self.H.size else H
            self.H = np.concatenate((self.H, H), axis=1) if self.H.size else H

            if DEBUG:
                print("normal_X size = %d, %d" % (len(normal_X), len(normal_X[0])))
                print("Synergy %d have %d component" % (index, opti_com))
                index += 1 
                #print(len(self.H))
                #print(self.H, model.components_)
                pass

        synergy, V, CPUtime, NRV, RRV = self.calfhals(flat_x, self.H, 30)
        #multiplot(synergy)
        #synergy = self.__synergyComputation(flat_x, self.H)
        normal_synergy, self.normal_synergy_range = self.__normalize(synergy)
        #multiplot(normal_synergy)
        
        if DEBUG:
            print("nflat_x size = %d, %d" % (len(nflat_x), len(nflat_x[0])))
            print("self.H size = %d, %d" % (len(self.H), len(self.H[0])))

        return self

    def transform(self, source):
        """
        Calculate synergy and have option for increase performance.

        Parameters
        ----------
        source : {array-like, sparse matrix} of shape (time, channel)
            data source to transform to synergy

        Returns
        -------
        nsynergy : returns nomallized transformed synergy.
        """
        assert len(source) != 0, "source is empty"

        #nsource, _ = self.__normalize(source, self.normal_X_range)
        #synergy = self.__synergyComputation(nsource, self.H)
        #nsynergy, self.normal_synergy_range = self.__normalize(synergy, self.normal_synergy_range)


        #synergy, _ = self.__normalize(tp(ans), self.normal_synergy_range)
        #return tp(synergy)

        U, V, CPUtime, NRV, RRV = self.calfhals(source, self.H, 30)
        nsynergy, self.normal_synergy_range = self.__normalize(U, self.normal_synergy_range)

        return nsynergy

    def calfhals(self, source, H, iter=30):
        """
        Calcualte synergy by minimize euclidean distance

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (time, channel)
            data source to transform to synergy

        U : array-like of shape (time, component)
            transform matrix according to NMF model

        V : array-like of shape (channel, component)
            transform matrix according to NMF model

        iter: interations

        Returns
        -------
        CPUtime: CPUtime taken for each update
        
        NRV    : Normalized Residual Value for each update
                     || X-UV^T || / ||X||

        RRV    : Relative Residual Value for each update
                    log_{10} (||X-U_{t}V_{t}|| / ||X-U_{0}V_{0}||)

        X = M (number of sample) x N (number of channel) matrix
        U = M (number of sample) x K (number of component) matrix
        V = N (number of channel) x K (number of component) matrix
        """
        
        X = source
        V = H
        component_num = H.shape[1]
        sample_num = len(source)

        from numpy import random
        U = abs(random.random([sample_num, component_num]))

        CPUtime = np.zeros(iter+1)
        NRV = np.zeros(iter+1)
        RRV = np.zeros(iter+1)
        eps = 1e-08

        from numpy import linalg as LA
        normX = LA.norm(X, 'fro')**2

        if DEBUG:
            print("normX:", normX)

        Ap0 = normX - 2*np.trace(dot(dot(tp(U), X), V)) + np.trace(dot(dot(tp(U), U), (dot(tp(V), V))))

        if DEBUG:
            print("Ap0:", Ap0)

        CPUtime[0] = 0
        NRV[0] = Ap0 / normX
        RRV[0] = 1

        import time
        import math
        for i in range(iter):
            t = time.time()

            A = dot(X, V)
            B = dot(tp(V), V)

            for j in range(component_num):
                tmp = (A[:, j] - dot(U, B[:, j]) + dot(U[:, j], B[j, j])) / B[j, j]
                tmp = [x if x > eps else eps for x in tmp]
                U[:, j] = tmp

            CPUtime[i]=time.time()-t
            Ap = normX - 2*np.trace(dot(dot(tp(U), X), V)) + np.trace(dot(dot(tp(U), U), (dot(tp(V), V))))
            NRV[i+1] = Ap / normX  

            try:
                RRV[i+1] = math.log10(Ap / Ap0)
            except ValueError:
                RRV[i+1] = 0

        return U, V, CPUtime, NRV, RRV

    def fasthals(self, X, U, V, iter=30):
        """
        Calcualte synergy by minimize euclidean distance

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (time, channel)
            data source to transform to synergy

        U : array-like of shape (time, component)
            transform matrix according to NMF model

        V : array-like of shape (channel, component)
            transform matrix according to NMF model

        iter: interations

        Returns
        -------
        CPUtime: CPUtime taken for each update
        
        NRV    : Normalized Residual Value for each update
                     || X-UV^T || / ||X||

        RRV    : Relative Residual Value for each update
                    log_{10} (||X-U_{t}V_{t}|| / ||X-U_{0}V_{0}||)

        X = M (number of sample) x N (number of channel) matrix
        U = M (number of sample) x K (number of component) matrix
        V = N (number of channel) x K (number of component) matrix
        """

        component_num = U.shape[1]
        #sample_num = X.shape[0]

        #from numpy import random
        #U = abs(random.random([sample_num, component_num]))

        CPUtime = np.zeros(iter+1)
        NRV = np.zeros(iter+1)
        RRV = np.zeros(iter+1)
        eps = 1e-08

        from numpy import linalg as LA
        normX = LA.norm(X, 'fro')**2

        if DEBUG:
            print("normX:", normX)

        Ap0 = normX - 2*np.trace(dot(dot(tp(U), X), V)) + np.trace(dot(dot(tp(U), U), (dot(tp(V), V))))

        if DEBUG:
            print("Ap0:", Ap0)

        CPUtime[0] = 0
        NRV[0] = Ap0 / normX
        RRV[0] = 1

        import time
        import math
        for i in range(iter):
            t = time.time()

            A = dot(X, V)
            B = dot(tp(V), V)

            for j in range(component_num):
                tmp = (A[:, j] - dot(U, B[:, j]) + dot(U[:, j], B[j, j])) / B[j, j]
                tmp = [x if x > eps else eps for x in tmp]
                U[:, j] = tmp

            A = dot(tp(X), U)
            B = dot(tp(U), U)

            for j in range(component_num):
                tmp = (A[:, j] - dot(V, B[:, j]) + dot(V[:, j], B[j, j])) / B[j, j]
                tmp = [x if x > eps else eps for x in tmp]
                V[:, j] = tmp


            CPUtime[i]=time.time()-t
            Ap = normX - 2*np.trace(dot(dot(tp(U), X), V)) + np.trace(dot(dot(tp(U), U), (dot(tp(V), V))))
            NRV[i+1] = Ap / normX  

            try:
                RRV[i+1] = math.log10(Ap / Ap0)
            except ValueError:
                RRV[i+1] = 0

        return U, V, CPUtime, NRV, RRV

    def __synergyComputation(self, source, synH):
        """
        Calculate synergy and have option for increase performance.

        Parameters
        ----------
        source : {array-like, sparse matrix} of shape (time, channel)
            data source to transform to synergy

        synH : array-like of shape (channel, component)
            transform matrix according to NMF model

        Returns
        -------
        synergy : returns transformed synergy.
        """
        assert len(source[0]) == len(synH), "source and synergy coefficient should have the same channel range"

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


    def __findoptimalcomp(self, X, y, max_components = 10, criteria = 0.9):
        """
        Find optimal number of component of NMF

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (channel, time)
            data source to transform to synergy

        y : {array-like, sparse matrix} of shape (channel, time)
            data needed as reference for reward function

        max_components : {int} max number of component, default 10

        criteria : {float} criteria value

        Returns
        -------
        max_cc_component : {int} number of component with max reward value or more than criteria
        """
        n_components = 2
        max_cc = 0
        max_cc_component = 0

        while n_components < max_components:
            n_components = n_components + 1

            X = np.array(X)

            channel_num = X.shape[1]
            sample_num = X.shape[0]

            from numpy import random
            U = abs(random.random([sample_num, n_components]))
            V = abs(random.random([channel_num, n_components]))

            U, V, CPUtime, NRV, RRV = self.fasthals(X, U, V, 1000)

            #multiplot(U)
            cc = self.__efficientindicator(U, y)
            if cc > criteria:
                return n_components
            if cc > max_cc:
                max_cc = cc
                max_cc_component = n_components
            
            print("cc = %f, n_components = %d" % (cc, n_components))

            if DEBUG:
                pass
                
        return max_cc_component

    def __efficientindicator(self, X, y):
        """
        Calcualte synergy by minimize euclidean distance

        Parameters
        ----------
        X : array-like of shape (time, channel)
            input data that need to find efficientcy

        y : array-like of shape (time, channel)
            correct data

        Returns
        -------
        cc : float
             maxmimum cc value

        X = M (number of sample) x N (number of channel) matrix
        y = M (number of sample) x N (number of channel) matrix
        """
        X = tp(X)
        y = tp(y)

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
                    print("x_channel, y_pos")
                    print("x_channel, y_neg")
                    pass
            cc.append(sum(accu_cc)/len(accu_cc))
            if DEBUG:
                #print(cc)
                pass
           
        return max(cc)

    def __normalize(self, data, channel_range = None):
        """
        Calculate synergy and have option for increase performance.

        Parameters
        ----------
        data : {array-like, sparse matrix} of shape (time, channel)
            data source to transform to synergy

        channel_range : array-like of tuple (min, max) with len(channel) 
            of each channel for normalization

        Returns
        -------
        normalized_data : {array-like, sparse matrix} of shape (channel, time) 
            normalized data

        channel_range : array-like of tuple (min, max) with len(channel)
            min and max value used in normalized_data data
        """
        data = tp(data)

        normalized_data = []
        if channel_range == None:
            channel_range = [(min(channel), max(channel)) for channel in data]

        assert len(data) == len(channel_range), "data and channel_range should have the same channel range: %d and %d" % (len(data), len(channel_range))


        for ich in range(len(data)):
            range_value = float(channel_range[ich][1]-channel_range[ich][0])
            if range_value == 0:
                normal_data = data[ich] - data[ich]
            else:
                normal_data = (data[ich] - channel_range[ich][0]) / range_value
            for inor in range(len(normal_data)):
                if normal_data[inor] < 0:
                    normal_data[inor] = 0.0
            normalized_data.append(normal_data)
        return tp(normalized_data), channel_range

    def NMF(self, data, syn_num):
        """
        data: time x channel
        syn_num : int
        """
        i, j, k, l = 0,0,0,0
        RR = 0.0
        V = data                # time * channe;
        M = len(data)           # time
        N = len(data[0])        # channel
        K = syn_num             # number of synergy (syn_num)

        eps = 1e-08
        W = np.random.rand(M, K)    # M * K
        H = np.random.rand(N, K)    # N * K

        mean = np.zeros(M , dtype=float)
        R2 = np.zeros(10000, dtype=float)

        for i in range(10000):

            VH = dot(V,H)           # time * channel x channel * syn_num -> time * syn_num
            HTH = dot(tp(H), H)

            for j in range(K):
                HTHs = np.zeros((K, 1), dtype=float)   # K * 1
                for k in range(K):
                    HTHs[k, 0] = HTH[k, j]
                WHTHs = dot(W, HTHs)    # M * 1
                for k in range(M):
                    temp = (VH[k, j] - WHTHs[k, 0] + W[k, j] * HTH[j, j]) / HTH[j, j]
                    W[k, j] = eps if temp <= eps else temp
            
            W, _, _ = self.normalize(W)

            VT = tp(V)              # channel * time (M * N)
            VTW = dot(VT, W)        # channel * time x time * syn_num -> channel * syn_num
            WTW = dot(tp(W), W)

            for j in range(K):
                WTWs = np.zeros((K, 1), dtype=float)   # K * 1
                for k in range(K):
                    WTWs[k, 0] = WTW[k, j]
                HWTWs = dot(H, WTWs)
                for k in range(N):
                    temp = (VTW[k, j] - HWTWs[k, 0] + H[k, j] * WTW[j, j]) / WTW[j, j]
                    H[k, j] = eps if temp <= eps else temp

            HT = tp(H)          # K * N
            WH = dot(W, HT)       # M * K x K * N -> channel * time (M * N)
            sse = 0.0
            sst = 0.0
            average = 0.0
            for j in range(M):
                for k in range(N):
                    sse += (V[j, k] - WH[j, k]) * (V[j, k] - WH[j, k])

            for j in range(M):
                average = 0.0
                for k in range(N):
                    average += V[j, k]
                average = average / N
                mean[j] = average
            
            for j in range(M):
                for k in range(N):
                    sst += (V[j, k] - mean[j]) * (V[j, k] - mean[j])

            R2[i] = 1 - sse / sst
            RR = R2[i]
            
            if i > 20:
                compare = 0
                for j in range(20):
                    temp = abs(R2[i - j] - R2[i - j - 1])
                    if temp < 1.0e-5:
                        compare += 1
                if compare == 20:
                    break

        W_result = W
        H_result = H
        Rsquare = RR

        return W, H, RR


if __name__ == "__main__":
    import os

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

    #filename = os. getcwd() + '/../data/train_data_09_17_2020_11_13_54.csv'
    filename = os. getcwd() + '/../data/train_data_11_25_2020_16_36_00.csv'
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

    nEMGdata, minEMGdata, maxEMGdata = array_normalize(EMGdata)

    # for i in range(len(data_X)):
    #     data_X[i] = np.array(data_X[i])
    # for i in range(len(data_Y)):
    #     data_Y[i] = np.array(data_Y[i])

    gripdata_X = np.concatenate((data_X[0], data_X[1], data_X[2]), axis=1)
    gripdata_Y = np.concatenate((data_Y[0], data_Y[1], data_Y[2]), axis=1)

    wristdata_X = np.concatenate((data_X[0], data_X[3], data_X[4]), axis=1)
    wristdata_Y = np.concatenate((data_Y[0], data_Y[3], data_Y[4]), axis=1)

    prosudata_X = np.concatenate((data_X[0], data_X[5], data_X[6]), axis=1)
    prosudata_Y = np.concatenate((data_Y[0], data_Y[5], data_Y[6]), axis=1)

    sum_x = [tp(gripdata_X), tp(wristdata_X), tp(prosudata_X)]
    sum_y = [tp(gripdata_Y), tp(wristdata_Y), tp(prosudata_Y)]

    syn = synergy()
    syn.fit(sum_x, sum_y)
    transform_y = syn.transform(EMGdata)
    multiplot(transform_y)
    #plot(EMGdata)
    #plot(transform_y)

    #ny, _, _ = array_normalize(transform_y)

    #multiplot(tp(ny))
    #multiplot(tp(tp(ny)[1:10])) 

    #transform_y = syn.transform(tp(gripdata_X))

    # syn = synergy()
    # syn.fit([EMGdata], [tp(Motiondata)])
    # transform_y = syn.fasthals(tp(np.array(nEMGdata)), syn.H, 30)
    # multiplot(transform_y)
    # transform_y = syn.transform(EMGdata)
    # multiplot(transform_y)


    print("Test realtime processing")
    y = [0]*len(EMGdata)
    for i in range(len(EMGdata)):
        y[i] = syn.transform([EMGdata[i]])[0]

    multiplot(y)

    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()

    regressor.fit(y, Motiondata)
    angle = regressor.predict(y)
    multiplot(angle)


    filename = os.getcwd() + '/../data/run_data_11_25_2020_13_23_57.csv'
    data = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                data.append([float(value) for value in row])
            except:
                print(row)

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

    transform_y = syn.transform(EMGdata)
    multiplot(transform_y)
    angle = regressor.predict(transform_y)
    multiplot(angle)