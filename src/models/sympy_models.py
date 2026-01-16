import sympy as sym

class OneStep:
    def __init__(self, measure=("so",)):
        k = sym.symbols("k")
        s1, s2, so = sym.symbols("s1 s2 so")

        self.p = [[k]]
        self.x = [[s1], [s2], [so]]
        self.u = []   
        self.w = []
        self.f = [[-k*s1*s2],
                  [-k*s1*s2],
                  [ k*s1*s2]]

        self.h = _select_outputs(self.x, measure)
        self.name = "one_step"

    def as_strikepy_dict(self):
        return {"p": self.p, "x": self.x, "u": self.u, "w": self.w, "f": self.f, "h": self.h}



class OneStepLatent:
    def __init__(self, measure=("so",)):
        k1, k2 = sym.symbols("k1 k2")
        s1, s2, s3, so = sym.symbols("s1 s2 s3 so")

        self.p = [[k1], [k2]]
        self.x = [[s1], [s2], [s3], [so]]
        self.u = []
        self.w = []
        self.f = [[-k1*s1*s1],
                  [-k2*s1*s1],
                  [ k1*s1*s2 - k2*s3],
                  [ k2*s2]]

        self.h = _select_outputs(self.x, measure)
        self.name = "one_step_latent"

    def as_strikepy_dict(self):
        return {"p": self.p, "x": self.x, "u": self.u, "w": self.w, "f": self.f, "h": self.h}
    

class TwoStepUni:
    def __init__(self, measure=("so",)):
        k1, k2 = sym.symbols("k1 k2")
        s1, s2, so = sym.symbols("s1 s2 so")

        self.p = [[k1], [k2]]
        self.x = [[s1], [s2], [so]]
        self.u = []
        self.w = []
        self.f = [[-k1*s1],
                  [ k1*s1 - k2*s2],
                  [ k2*s2]]

        self.h = _select_outputs(self.x, measure)
        self.name = "two_step_uni"

    def as_strikepy_dict(self):
        return {"p": self.p, "x": self.x, "u": self.u, "w": self.w, "f": self.f, "h": self.h}
    

class ThreeStepUni:
    def __init__(self, measure=("so",)):
        k1, k2, k3 = sym.symbols("k1 k2 k3")
        s1, s2, s3, so = sym.symbols("s1 s2 s3 so")

        self.p = [[k1], [k2], [k3]]
        self.x = [[s1], [s2], [s3], [so]]
        self.u = []
        self.w = []
        self.f = [[-k1*s1],
                  [ k1*s1 - k2*s2],
                  [ k2*s2 - k3*s3],
                  [ k3*s3]]

        self.h = _select_outputs(self.x, measure)
        self.name = "three_step_uni"

    def as_strikepy_dict(self):
        return {"p": self.p, "x": self.x, "u": self.u, "w": self.w, "f": self.f, "h": self.h}

def _select_outputs(x, names):
    """
    x: list-of-lists state vector [[s1],[s2],...]
    names: iterable of strings OR SymPy Symbols, e.g. ("s1","so") or (s1, so)

    Returns h in StrikePy format: [[sym1],[sym2],...]
    """
    # flatten x to symbols
    x_syms = [row[0] for row in x]
    by_name = {s.name: s for s in x_syms}

    out = []
    for n in names:
        if isinstance(n, sym.Symbol):
            out.append([n])
        else:
            # assume string
            if n not in by_name:
                raise ValueError(f"Requested output '{n}' not in states: {list(by_name.keys())}")
            out.append([by_name[n]])
    return out

class OneStepCatalyst:
    def __init__(self, measure=("so",)):
        k = sym.symbols("k")
        s1, s2, so = sym.symbols("s1 s2 so")

        self.p = [[k]]
        self.x = [[s1], [s2], [so]]
        self.u = []
        self.w = []
        self.f = [[0],
                  [-k*s1*s2],
                  [ k*s1*s2]]

        self.h = _select_outputs(self.x, measure)
        self.name = "one_step_catalyst"

    def as_strikepy_dict(self):
        return {"p": self.p, "x": self.x, "u": self.u, "w": self.w, "f": self.f, "h": self.h}

class TwoStepCatalyst:
    def __init__(self, measure=("so",)):
        k1, k2 = sym.symbols("k1 k2")
        s1, s2, s3, s4, so = sym.symbols("s1 s2 s3 s4 so")

        self.p = [[k1], [k2]]
        self.x = [[s1], [s2], [s3], [s4], [so]]
        self.u = []
        self.w = []
        self.f = [[-k1*s1*s2 + k2*s3*s4],
                  [-k1*s1*s2],
                  [k1*s1*s2 - k2*s3*s4],
                  [-k2*s3*s4],
                  [ k2*s1*s2]]
    
        self.h = _select_outputs(self.x, measure)
        self.name = "two_step_catalyst"

    def as_strikepy_dict(self):
        return {"p": self.p, "x": self.x, "u": self.u, "w": self.w, "f": self.f, "h": self.h}