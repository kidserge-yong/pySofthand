import csv
import numpy as np
from sklearn.decomposition import NMF
import pandas as pd
import matplotlib.pyplot as plt 
import math

inv = lambda n: np.linalg.inv(n)
tp = lambda n: np.transpose(n)
dot = lambda x, y: np.dot(x,y)

class Synergy:

    data = np.array([])
    maxmin = []
    init_method = 'random'
    solver = 'mu'
    max_iter = 1000
    model = None
    A = None
    W = None
    H = None
    
    def __init__(self):
        print("Synergy: %s has been built." % __name__)
        pass
        
    def test(self, text: str ="") -> str:
        
        return(1)

    def input(self, pathTocsv: str = r".\data\NORM.csv"):
        # with open(pathTocsv) as csvfile:
        #     tem = list(csv.reader(csvfile))

        #     for row in range(len(tem)):
        #         rowlen = len(tem[row])
        #         for elem in range(rowlen):
        #             tem[row][elem] = float(tem[row][elem])
        # self.data = np.array(tem)

        df=pd.read_csv(pathTocsv, sep=',',header=None)
        self.data = df.values

        
        tem = []
        for elem in self.data:
            tem += [[min(elem), max(elem)]]
        self.maxmin = tem

    def normalize(self, data, array_min=None, array_max=None):
        """
        data[[time] x channel]: data of each channel and time
        """
        data = tp(data)
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
            # for inor in range(len(normal_data)):
                # if normal_data[inor] < 0:
                #     normal_data[inor] = 0.0
            ans.append(normal_data)
        return tp(ans), min_ans, max_ans

    def findOptimalCompNum(self, target=None, max_components=0, rsqure_criteria=0.9, tol=0.0001):
        ''' Commecnt on how to use '''
        n_components = 1
        rsqure_list = []
        while n_components <= max_components:
            model = NMF(n_components = n_components, init = self.init_method, solver = self.solver, max_iter = self.max_iter)
            W = model.fit_transform(target)
            H = model.components_

            # W, H, model = self.NMF(target, n_components)
            # H = tp(H)

            predict = np.dot(W, H)
            rsqure = self.rsqure(target, predict)
            rsqure_list.append(rsqure)
            

            if rsqure > rsqure_criteria:
                print("finded rsqure more than %f with n_components = %d and rsqure = %f" % (rsqure_criteria, n_components, rsqure))
                break
            
            if len(rsqure_list) > 2:
                if rsqure - rsqure_list[-2] < tol:
                    print("changes rsqure less than %f with n_components = %d and rsqure = %f" % (tol, n_components, rsqure))
                    break

            n_components = n_components + 1

        if n_components >= max_components:
            print("n_components more than max component %d with n_components = %d and rsqure = %f" % (max_components, n_components, rsqure))
        
        return rsqure_list

    

    def trainModel(self, sourceSignal, n_components = 0):
        '''
        train model according to input or properties of data
        sourceSignal should be in dimention of (time x input)
        '''

        model = NMF(n_components = n_components, init = self.init_method, solver = self.solver, max_iter = self.max_iter)
        W = model.fit_transform(sourceSignal)
        H = model.components_
        H = tp(H)
        
        # W, H, model = self.NMF(sourceSignal, n_components)
        
        H, _, _ = self.normalize(H)

        self.model = model
        self.A = sourceSignal
        self.W = W
        self.H = H


    def getSynergy(self, sourceSignal):
        '''
        get Synergy from synergy.H variable and return synergy in dimention of (time x n_component)
        sourceSignal should be in dimention of (time x input)
        '''

        dot = lambda x, y: np.dot(x,y)
        #synergy = dot(sourceSignal, self.getConvertMetric())
        synergy = dot(sourceSignal, self.H)
        return synergy

        
    def getConvertMetric(self):
        '''
        get Synergy from synergy.H variable in dimention of (input x n_component)
        '''
        if self.H is None:
            print('Model empty please trainModel first. Return -1')
            return -1

        inv = lambda n: np.linalg.inv(n)
        tp = lambda n: np.transpose(n)
        dot = lambda x, y: np.dot(x,y)

        # HHt = dot(self.H,tp(self.H))
        # invHHt = inv(HHt)
        # syn_co = dot(tp(self.H), invHHt)

        invHtH = inv(dot(tp(self.H),self.H))
        coef = dot(self.H, invHtH)
        coef, _ , _ = self.normalize(coef)
        return coef


    def rsqure(self, target, predict):
        '''returns R squared errors (model vs actual)'''
        return 1-self.sse(target,predict)/self.sst(target)

    def sse(self, target, predict):
        '''returns sum of squared errors (model vs actual)'''
        squared_errors = (target - predict) ** 2
        return np.sum(np.sum(squared_errors))
        
    def sst(self, target):
        '''returns total sum of squared errors (actual vs avg(actual))'''
        avg_y = np.mean(target)
        squared_errors = (target - avg_y) ** 2
        return np.sum(np.sum(squared_errors))

    def synmagcomputation(self, iemg, syncoof):
        e = 0.01
        invStS = inv(np.dot(tp(syncoof), syncoof))
        invStSSt = np.dot(invStS, tp(syncoof))
        ans = []
        for i in range(len(iemg)):
            eps = e
            Tx = tp(np.dot(invStSSt, tp(iemg[i])))
            count = 0   
            # while((Tx < -eps).any()):
            #     minus = np.where((Tx < -eps)==True)
            #     plus = np.where((Tx < -eps)==False)
            #     minus = minus[0]
            #     plus = plus[0]

            #     # delete channel that is minus
            #     new_S = np.delete(syncoof, minus, 1)
            #     cut = np.array([0.0]*len(Tx))

            #     num = np.dot(syncoof[:, minus], tp(abs(Tx[minus])))
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
            #     #print(Tx)
            ans.append(Tx)
            #print(Tx)
        return ans

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



if __name__ == "main":
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
        return tp(ans), min_ans, max_ans

    def SynMagComputation(iemg, synH):
        e = 0.001
        invStS = inv(np.dot(tp(syncoof), syncoof))
        invStSSt = np.dot(invStS, tp(syncoof))
        ans = []
        for i in range(len(iemg)+1):
            eps = e
            Tx = tp(np.dot(invStSSt, tp(iemg[i])))
            count = 0   
            while((Tx < -eps).any()):
                minus = np.where((Tx < -eps)==True)
                plus = np.where((Tx < -eps)==False)
                minus = minus[0]
                plus = plus[0]

                # delete channel that is minus
                new_S = np.delete(syncoof, minus, 1)
                cut = np.array([0.0]*len(Tx))

                num = np.dot(syncoof[:, minus], tp(abs(Tx[minus])))
                num = np.dot(tp(new_S), num)
                den = inv(np.dot(tp(new_S),new_S))
                test = np.dot(den, num)

                for j, item in enumerate(plus):
                    cut[item] = test[j]
                    #print(i, item)

                # for j, item in enumerate(minus):
                #     Tx[item] = 0.0
                    #print(i, item)

                Tx = Tx - cut
                count += 1
                eps = e*10**(np.floor(math.log10(count)))
            #     #print(Tx)
            ans.append(Tx)
            #print(Tx)
        return ans

    import os

    filename = os. getcwd() + '/data/train_data_09_07_2020_14_17_34.csv'
    data = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                data.append([float(value) for value in row])
            except:
                #print(row)
                pass

    EMGdata = [item[:-4] for item in data]
    Motiondata = [item[-4:] for item in data]

    data_X = []
    data_Y = []
    tem_X = []
    tem_Y = []
    pre_j = Motiondata[0]

    for i,j in zip(EMGdata,Motiondata):
        #print(j)
        if j != pre_j:
            data_X.append(tem_X)
            data_Y.append(tem_Y)
            tem_X = []
            tem_Y = []
        tem_X.append(i)
        tem_Y.append(j)
        pre_j = j

    nEMGdata, minEMGdata, maxEMGdata = array_normalize(tp(EMGdata))

    gripdata_X = data_X[0] + data_X[1] + data_X[3]
    gripdata_X, tem2 , tem3 = array_normalize(tp(gripdata_X), minEMGdata, maxEMGdata)
    gripdata_Y = data_Y[0] + data_Y[1] + data_Y[3]

    wristdata_X = data_X[0] + data_X[5] + data_X[7]
    wristdata_X, tem2 , tem3 = array_normalize(tp(wristdata_X), minEMGdata, maxEMGdata)
    wristdata_Y = data_Y[0] + data_Y[5] + data_Y[7]

    synGripOpen = Synergy()
    synWrist = Synergy()

    opti_syn = synGripOpen.findOptimalCompNum(gripdata_X, max_components=20, rsqure_criteria=0.85, tol=0.00001)
    synGripOpen.trainModel(gripdata_X, n_components = len(opti_syn))
    opti_syn = synGripOpen.findOptimalCompNum(wristdata_X, max_components=20, rsqure_criteria=0.85, tol=0.00001)
    synWrist.trainModel(wristdata_X, n_components = len(opti_syn))

    synGripOpencoof = synGripOpen.getConvertMetric()
    synWristcoof = synWrist.getConvertMetric()

    syn_grip = synGripOpen.getSynergy(gripdata_X)
    syn_wrist = synWrist.getSynergy(wristdata_X)

    new_syncoof = [
    [0.0140609215712943	,0.111085379838387	,0.825145053751431	,0.0292683058095186	,0.128234340070876	,0.470464645535291],
    [3.27120157021131e-09	,0.124300255774238	,0.888475796926454	,2.64429696601194e-09	,5.86076917979398e-09	,1],
    [0.0487976281715278	,0.0773971117154516	,0.342177375869361	,2.64429696601194e-09	,0.0777075870667463	,0.862436171418091],
    [0.284081486311818	,0.425758055232655	,0.907832587277067	,0.273723538982002	,0.665148297445691	,0.793313367813018],
    [0.196678167169097	,0.580479132664513	,0.0711496779732134	,0.177459784540193	,1	,0.236487103485887],
    [3.27120157021131e-09	,1	,0.122409594530106	,2.64429696601194e-09	,0.380321286355776	,0.107708914390312],
    [0.243604907083157	,0.723530736834703	,0.0181440795282096	,0.230719251501190	,0.160558216510520	,0.149902470864350],
    [0.736774688782458	,0.419190758500866	,0.361141218155819	,0.669785672911855	,0.619127851112348	,0.0503491593726449],
    [0.0157160641436323	,0.0625627191430681	,0.912389897940566	,0.0350482608585051	,0.103170714803817	,0.322145161526901],
    [3.27120157021131e-09	,0.0148949934770332	,0.905902320867239	,2.64429696601194e-09	,0.0132001324229941	,0.593461064066232],
    [0.0129628385962094	,0.0818725086417938	,0.827455857140013	,2.64429696601194e-09	,0.231792539279453	,0.932310349129326],
    [0.201672060802906	,0.323898981987285	,0.613589679422028	,0.138977131576379	,0.604089293560023	,0.703228274550494],
    [0.122354299191554	,0.795144986278498	,0.113055208416874	,0.117442375354009	,0.899652505612173	,0.199652374576423],
    [0.167236113214762	,0.748212435155281	,9.45326206990387e-09	,0.173983726596382	,0.920044122452085	,0.0924680599060946],
    [0.653680900847857	,0.619490315186549	,0.960698770555061	,0.607602964134838	,0.449468793443311	,0.985258220003516],
    [0.287407532842042	,0.860124993661778	,1	,0.276513055701354	,0.808816789928840	,0.655553154020699],
    [0.0244685998581473	,0.0867347727363793	,0.981459283473166	,0.0440183018876841	,0.115640258368531	,0.467574822124334],
    [0.0207183691784577	,0.0820164790248052	,0.988702958303255	,0.0257471435858758	,0.0585136598477129	,0.628142491853624],
    [0.206303548230586	,0.0478645050514794	,0.852160190663243	,0.201749540517031	,0.341515549560709	,0.349395114648657],
    [0.158423638590710	,0.811515842557380	,0.446804573710549	,0.0997387817207516	,0.697897584528614	,0.677214765689282],
    [0.0442381599282066	,0.973180771564477	,0.140465621198119	,0.0263320041650781	,0.543526600162520	,0.503790620830420],
    [0.122976961259511	,0.383714660763651	,0.107016602400256	,0.100721088602650	,0.982468134431059	,0.0568057475898209],
    [0.509073405953853	,0.808917943274034	,0.813513845329649	,0.433901203375019	,0.121804122630817	,0.853815774477121],
    [0.839031784067931	,0.737978829907525	,0.375115878482288	,0.756543759444424	,0.262471823608104	,0.368264094381601],
    [0.000141369248342721	,0.0861006164655445	,0.849159190551611	,0.0396693516571075	,0.143276180716828	,0.364788339462250],
    [0.0116454541876350	,0.0294453011405237	,0.957155500326141	,0.0237083228900443	,0.00750813052348785	,0.563204182758640],
    [0.0143486413888316	,6.45033856626817e-09	,0.799086128811007	,0.0246726339165605	,0.465906256720561	,0.264260677191604],
    [0.0937962550724693	,0.719854617478338	,0.700458076529478	,0.0623606566267841	,0.708658762209505	,0.536350148281942],
    [0.128010666370161	,0.889506594283852	,0.0889078896788193	,0.110750879855163	,0.380798408505199	,0.324891002463008],
    [0.114084373893289	,0.519379032714114	,0.224633068641459	,0.0932997935997830	,0.921924457804606	,0.118441311330654],
    [0.949763663638742	,0.626231293812943	,0.909421454345308	,0.875709543038299	,0.457484673978549	,0.891040878324926],
    [1	,0.193921412311705	,0.201006065820555	,1	,0.356570685693097	,0.115838330401218]
    ]

    syncoof = np.array(new_syncoof)
    syncoof = np.concatenate((synGripOpen.H, synWrist.H), axis=1)

    syncoof, tem2 , tem3 = array_normalize(tp(syncoof))

    fig, axs = plt.subplots(1,2)
    #axs[0].plot(gripdata_X)
    axs[0].plot(syncoof)
    #axs[1].plot(wristdata_X)
    axs[1].plot(new_syncoof)
    plt.show()

    Dat = SynMagComputation(nEMGdata, syncoof)
    nDat, tem2 , tem3 = array_normalize(tp(Dat))

    multiplot(Dat)

