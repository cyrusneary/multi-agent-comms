import pylab as pl
import numpy as np

a = np.array([[0, 2.06]])
pl.figure(figsize=(30, 1.0))
img = pl.imshow(a, cmap="Oranges")
pl.gca().set_visible(False)
cax = pl.axes([0.1, 0.2, 0.8, 0.6])
pl.colorbar(orientation="horizontal", cax=cax)
pl.savefig("colorbar.pdf")