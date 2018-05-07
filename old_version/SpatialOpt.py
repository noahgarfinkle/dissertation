# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
c = [-1, 4]
A = [[-3, 1], [1, 2]]
b = [6, 4]
x0_bounds = (None, None)
x1_bounds = (-3, None)
from scipy.optimize import linprog
res = linprog(c, A_ub=A, b_ub=b, bounds=(x0_bounds, x1_bounds),
              options={"disp": True})
print(res)


"""
Implements a Stackelberg Game where player 1 chooses to invest resources to
defend a node and player 2 then invests resources to attack a node.  If player
1 is defending the attacked node, damage is mitigated.  Utility is scored by
routing capabilities
"""
def competitiveRouting():
    return 0
