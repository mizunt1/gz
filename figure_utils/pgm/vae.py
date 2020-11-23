import daft
from matplotlib import rc

rc("font", family="serif", size=12)
rc("text", usetex=True)


pgm = daft.PGM()
pgm.add_node("z", r"z", 3, 3, aspect=1)
pgm.add_node("x", r"x", 3, 2, aspect=1, observed=True)
pgm.add_node("phi", r"$\theta$",4,3 )
pgm.add_edge("z", "x")
pgm.add_edge("phi", "x")
# x start y start x len y llen
pgm.add_plate([3-0.6, 1.45, 1.2, 2.1], label="N")
pgm.add_node("z2", r"z", 3+2.5, 2, aspect=1)
pgm.add_node("x2", r"x", 3+2.5, 3, aspect=1, observed=True)
pgm.add_node("theta", r"$\phi$", 3+3.5, 3, aspect=1)

pgm.add_edge("x2", "z2")
pgm.add_edge("theta", "z2")
# x start y start x len y llen
pgm.add_plate([3-0.6+2.5, 1.45, 1.2, 2.1], label="N")

pgm.render()
pgm.savefig("vae.pdf")
pgm.savefig("vae.png", dpi=150)
