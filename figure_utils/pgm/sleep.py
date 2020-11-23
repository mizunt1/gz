import daft
from matplotlib import rc

rc("font", family="serif", size=6)
rc("text", usetex=True)


pgm = daft.PGM()
pgm.add_node("meal", r"meal", 2.5, 3, aspect=1)
pgm.add_node("slept", r"slept", 1, 3, aspect=1)
pgm.add_node("mistakes", r"errors", 1.75, 2.2, aspect=1, observed=True)
pgm.add_node("assignments", r"work due", 1, 4, aspect=1)
pgm.add_edge("meal", "mistakes", xoffset=-0.3)
pgm.add_edge("slept", "mistakes", xoffset=0.3)
pgm.add_edge("assignments", "slept")
pgm.render()
pgm.savefig("wordy.pdf")
pgm.savefig("wordy.png", dpi=150)
