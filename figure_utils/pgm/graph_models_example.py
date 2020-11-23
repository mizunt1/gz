import daft
from matplotlib import rc

rc("font", family="serif", size=12)
rc("text", usetex=True)


pgm = daft.PGM()
pgm.add_node("z", r"z", 2.5, 3, aspect=1)
pgm.add_node("x", r"x", 1.5, 3, aspect=1)
pgm.add_node("y", r"y", 2, 4, aspect=1)
pgm.add_edge("y", "z", xoffset=-0.3)
pgm.add_edge("y", "x", xoffset=0.3)

pgm.render()
pgm.savefig("gm_example.pdf")
pgm.savefig("gm_example.png", dpi=150)
