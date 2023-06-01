import anfis
import membership.mfDerivs
import membership.membershipfunction
import numpy
import matplotlib.pyplot as plt

# numpy.loadtxt('c://Python_fiddling//myProject//MF//trainingSet.txt',usecols=[1,2,3])
ts = numpy.load(
    "/home/estebanp/fuzzy-control/anfis/anfis/datos.npy")

# Normalize
ts = ts / ts.max(axis=0)

plt.subplot(4, 1, 1)
plt.plot(ts[:, 0])
plt.subplot(4, 1, 2)
plt.plot(ts[:, 1])
plt.subplot(4, 1, 3)
plt.plot(ts[:, 2])
plt.subplot(4, 1, 4)
plt.plot(ts[:, 3])
plt.show()

X = ts[:, 0:2]
Y = ts[:, 2]


mf = [[['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': 0.3, 'sigma': 1}], ['gaussmf', {'mean': 0.6, 'sigma': 1.}], ['gaussmf', {'mean': 1., 'sigma': 1.}]],
      [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': 0.3, 'sigma': 1}], ['gaussmf', {'mean': 0.6, 'sigma': 1.}], ['gaussmf', {'mean': 1., 'sigma': 1.}]]]

mfc = membership.membershipfunction.MemFuncs(mf)


anf = anfis.ANFIS(X, Y, mfc)

anf.trainHybridJangOffLine(epochs=20)

print("Plotting errors")
anf.plotErrors()
print("Plotting results")
anf.plotResults()
