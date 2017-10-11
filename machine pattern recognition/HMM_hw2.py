'''
Xin Wen    xwen4@binghamton.edu
Consider the use of hidden Markov models for classifying sequences of four visible
states, A-D. Train two hidden Markov models, each consisting of three hidden states
(plus a null initial state and a null final state), fully connected, with the following data.
Assume that each sequence starts with a null symbol and ends with an end null
symbol (not listed).
a. Print out the full transition matrices for each of the models.
b. Assume equal prior probabilities for the two models and classify each of the
following sequences: ABBBCDDD, DADBCBAA, CDCBABA, and
ADBBBCD.
c. As above, classify the test pattern BADBDCBA. Find the prior probabilities
for your two trained models that would lead to equal posteriors for your two
categories when applied to this pattern.
'''

import numpy as np

V1 = ['AABBCCDD', 'ABBCBBDD', 'ACBCBCD', 'AD', 'ACBCBABCDD', 'BABAADDD', 'BABCDCC', 'ABDBBCCDD', 'ABAAACDCCD', 'ABD']
V2 = ['DDCCBBAA', 'DDABCBA', 'CDCDCBABA', 'DDBBA', 'DADACBBAA', 'CDDCCBA', 'BDDBCAAAA', 'BBABBDDDCD', 'DDADDBCAA', 'DDCAAA']

# Load data
def initdata(V):
    # Assuming A, B, C, D correspond to state 0,1,2,3  null state 4
    output = []
    for w in V:
        temp = [4]  # initial and last observation
        for c in w:
            temp.append(ord(c)-65)
        temp.append(4)
        output.append(temp)
    return output
V1, V2 = map(initdata, [V1, V2])

class hmm(object):
    def __init__(self, dataset):

        self.data = np.array(dataset)
        self.l_v = 5                                     # number of visible states + null state
        self.l_w = 5                                     # number of hidden states, we know it's 3 + 2 (initial/final)

    def hmm_train(self):
        def r():                                                    # break symmetry
            return np.random.rand()*0.01
        a = np.array([[0, 0.25+r(), 0.25+r(), 0.25+r(), 0.25+r()],  # any state won't go to initial state
                    [0, 0.25+r(), 0.25+r(), 0.25+r(), 0.25+r()],    # final state will only go to final state
                    [0, 0.25+r(), 0.25+r(), 0.25+r(), 0.25+r()],
                    [0, 0.25+r(), 0.25+r(), 0.25+r(), 0.25+r()],
                    [0, 0, 0, 0, 1]])
        b = np.array([[0, 0, 0, 0, 1],                              # init and final state always lead to null
                      [0.25+r(), 0.25+r(), 0.25+r(), 0.25+r(), 0],  # the other invisible state will hve uniform
                      [0.25+r(), 0.25+r(), 0.25+r(), 0.25+r(), 0],  # possibility to the 4 visible state
                      [0.25+r(), 0.25+r(), 0.25+r(), 0.25+r(), 0],
                      [0, 0, 0, 0, 1]])
        threhold = 0.001
        cost = float('inf')
        iteration = 0
        while cost > threhold:
            iteration += 1
            Gamma = []
            Xi = []
            for V in self.data:
                alpha, beta, gamma, xi = self.forward_backward(a, b, V)
                Gamma.append(gamma)
                Xi.append(xi)
            a_nxt, b_nxt = self.update(a[:], b[:], Gamma, Xi, self.data)
            cost = max(np.abs(b_nxt - b).max(), np.abs(a_nxt - a).max())
            # print('Current a: {} \n Current b: {}'.format(a, b))
            # print('Current a: {} \n Current b: {}'.format(a, b))
            print('Current iteration: {},Cost: {}'.format(iteration, cost))
            a = a_nxt[:]
            b = b_nxt[:]
        self.a = a
        self.b = b
        print('Total iteration: {}, a: {} \n Current b: {}'.format(iteration, a, b))

    def compare(self, a1, b1, a2, b2, V):
        alpha1 = self.forward(a1, b1, V)
        alpha2 = self.forward(a2, b2, V)
        cum1 = np.sum(alpha1[:,-1])
        cum2 = np.sum(alpha2[:,-1])
        # print(alpha1, alpha2)
        if cum1>cum2:
            print('cum1:{} > cum2:{}, so {} belongs to w1'.format(cum1, cum2, V))
        elif cum1<cum2:
            print('cum1:{} < cum2:{}, so {} belongs to w2'.format(cum1, cum2, V))
        else:
            print('cum1 and cum2 have same possibillity')
        return cum1, cum2


    def forward(self, a, b, V):
        alpha = np.zeros(shape=(self.l_w, len(V)))            # alpha(i, t)  for P(V.T), final state
        alpha[:, 0] = np.array([1,0, 0, 0, 0])
        for t in range(len(V)-1):                          # length of observation
            for i in range(self.l_w):                      # i next layer (t+1)
                cum = 0
                for ii in range(self.l_w):                 # i current layer (t)
                    cum += alpha[ii,t] * a[ii, i] * b[i,V[t+1]]
                alpha[i, t+1] = cum
        # print('alp', alpha)
        return alpha

    def backward(self, a, b, V):
        beta = np.zeros(shape=(self.l_w, len(V)))  # for P(V.T), initial state
        beta[:, -1] = np.array([1, 1, 1, 1, 1])
        for t in range(0, len(V)-1)[::-1]:
            for i in range(self.l_w):  # cur(t) layer
                cum = 0
                for ii in range(self.l_w): # t+1 layer
                    cum += beta[ii, t+1] * a[i,ii] * b[ii, V[t+1]]
                beta[i, t] = cum
        # print('beta', beta)
        return beta

    def forward_backward2(self, a, b, V):
        alpha = self.forward(a, b, V)
        beta = self.backward(a, b, V)
        gamma = np.zeros((self.l_w, self.l_w, self.l_v, len(V)))


    def forward_backward(self, a, b, V):
        alpha = self.forward(a, b, V)
        beta = self.backward(a, b, V)
        gamma = np.zeros((self.l_w, len(V)))
        kai = np.zeros((self.l_w, self.l_w, len(V)-1))
        for t in range(len(V)):
            for i in range(self.l_w):
                gamma[i, t] = alpha[i, t] * beta[i, t]
            if sum(gamma[:,t]) != 0:
                gamma[:, t] = gamma[:, t] / sum(gamma[:, t])    # normalize
        for t in range(len(V)-1):
            for i in range(self.l_w):
                for j in range(self.l_w):
                    kai[i, j, t] = alpha[i, t] * a[i, j] * b[j, V[t+1]] * beta[j, t+1]
            kai[:,:,t] = kai[:,:,t] / np.sum(kai[:,:,t])
        return alpha, beta, gamma, kai

    def update(self, a, b, Gamma, Kai, V_all):
        # made mistakes here
        aa = np.copy(a)
        bb = np.copy(b)
        for i in range(0, self.l_w-1):
            for j in range(1, self.l_w):
                num = 0.001        # avoid zero division
                den = 0.001
                for sample in range(len(V_all)):
                    gamma = Gamma[sample]
                    kai = Kai[sample]
                    nu, de = np.sum(kai[i, j, 1:len(V_all[sample])-1]), np.sum(gamma[i, 1:len(V_all[sample])-1])
                    num += nu
                    den += de
                    aa[i, j] = num / den
                aa[i,:] = aa[i,:] / sum(aa[i,:])
        for j in range(1, self.l_w-1):
            for k in range(self.l_v-1):
                num = 0.0001
                den = 0.0001
                for sample in range(len(V_all)):
                    gamma = Gamma[sample]
                    kai = Kai[sample]
                    # print(V_all)
                    tk = [iii for iii in range(len(V_all[sample])) if V_all[sample][iii]==k]
                    num += np.sum(gamma[j, tk])
                    den += np.sum(gamma[j, :])
                bb[j, k] = num / den

                bb[j,:] = bb[j,:] / sum(bb[j,:])
        # print('###', aa)
        return aa, bb

# Part a
h1 = hmm(V1)
h1.hmm_train()
h2 = hmm(V2)
h2.hmm_train()
# Part b
print('Part B')
VV = [[4,0,1,1,1,2,3,3,3], [4,3,0,3,1,2,1,0,0], [4,2,3,2,1,0,1,0], [4,0,3,1,1,1,2,3]]
for v in VV:
    h2.compare(h1.a, h1.b, h2.a, h2.b, v)
# Part c
v = [4,1,0,3,1,3,2,1,0]
print('Part C')
cum1, cum2 = h2.compare(h1.a, h1.b, h2.a, h2.b, v)
print('Prior probability of w1', (cum2)/(cum1+cum2),'\n','Prior probability of w2', (cum1)/(cum1+cum2) )







