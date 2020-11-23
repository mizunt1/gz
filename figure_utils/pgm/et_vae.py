import daft
from matplotlib import rc

rc("font", family="serif", size=12)
rc("text", usetex=True)


pgm = daft.PGM()
# x, y
pgm.add_node("z", r"z", 3, 3, aspect=1)

pgm.add_node("x", r"x", 3, 2, aspect=1, observed=True)
pgm.add_node("phi", r"$\theta$",4,2.5 )
pgm.add_edge("z", "x")
pgm.add_edge("phi", "x")
# x start y start x len y llen
pgm.add_plate([3-0.7, 1.45, 1.4, 2.1], label="N")

pgm.add_node("z2", r"z", 3+2.5, 2, aspect=1)
pgm.add_node("x2", r"x", 3+2.5 -0.3, 3, aspect=1, observed=True)
pgm.add_node("theta", r"$\phi$", 3+3.5, 2.5, aspect=1)
pgm.add_node("pose", r"$\theta_{p}$", 3+2.5+0.3, 3, aspect=1)
pgm.add_edge("x2", "z2")
pgm.add_edge("pose", "z2")
pgm.add_edge("theta", "z2")

# x start y start x len y llen
pgm.add_plate([3-0.7+2.5, 1.45, 1.4, 2.1], label="N")

pgm.render()
pgm.savefig("et_vae.pdf")
pgm.savefig("et_vae.png", dpi=150)
