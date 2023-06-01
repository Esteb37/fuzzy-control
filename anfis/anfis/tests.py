import anfis
import membership.mfDerivs
import membership.membershipfunction
import numpy
import matplotlib.pyplot as plt

# numpy.loadtxt('c://Python_fiddling//myProject//MF//trainingSet.txt',usecols=[1,2,3])
ts = numpy.load(
    "C:/Users/esteb/OneDrive/Documents/Escuela/Robótica/Fuzzy/wang-mendel/datos.npy")[:1000]
plt.subplot(3, 1, 1)
plt.plot(ts[:, 0])
plt.subplot(3, 1, 2)
plt.plot(ts[:, 1])
plt.subplot(3, 1, 3)
plt.plot(ts[:, 3])
plt.show()

X = ts[:, 0:2]
Y = ts[:, 3]


mf = [[['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}], ['gaussmf', {'mean': -4., 'sigma': 10.}], ['gaussmf', {'mean': -7., 'sigma': 7.}]],
      [['gaussmf', {'mean': 1., 'sigma': 2.}], ['gaussmf', {'mean': 2., 'sigma': 3.}], ['gaussmf', {'mean': -2., 'sigma': 10.}], ['gaussmf', {'mean': -10.5, 'sigma': 5.}]]]

mfc = membership.membershipfunction.MemFuncs(mf)


anf = anfis.ANFIS(X, Y, mfc)

anf.trainHybridJangOffLine(epochs=20)

print(round(anf.consequents[-1][0], 6))
print(round(anf.consequents[-2][0], 6))
print(round(anf.fittedValues[9][0], 6))
if round(anf.consequents[-1][0], 6) == -5.275538 and round(anf.consequents[-2][0], 6) == -1.990703 and round(anf.fittedValues[9][0], 6) == 0.002249:
    print('test is good')

print("Plotting errors")
anf.plotErrors()
print("Plotting results")
anf.plotResults()
