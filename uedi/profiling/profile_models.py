
class DEP(object):
    def __init__(self, lhs, rhs=[]):
        if lhs == ['']:
            lhs = []
        self.lhs = lhs
        self.rhs = rhs

    def __hash__(self):
        return hash((tuple(self.lhs), tuple(self.rhs)))

    def __eq__(self, other):
        return ((set(self.lhs), set(self.rhs)) == (set(other.lhs), set(other.rhs)))

    def __le__(self, other):
        return ((set(self.lhs) <= set(other.lhs)) and (set(self.rhs) == set(other.rhs)))

    def __ge__(self, other):
        return ((set(other.lhs) <= set(self.lhs)) and (set(self.rhs) == set(other.rhs)))

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_human_readable_dep(self, columns_map):

        lhs = [columns_map[str(lhs_item)] for lhs_item in self.lhs]
        rhs = [columns_map[str(rhs_item)] for rhs_item in self.rhs]
        return [lhs, rhs]


class FD(DEP):
    def __str__(self):
        return str(self.lhs) + " -> " + str(self.rhs)


class IND(DEP):
    def __str__(self):
        return str(self.lhs) + " ⊆ " + str(self.rhs)


class ORD(DEP):
    def __init__(self, lhs, rhs, order_type, comp):
        DEP.__init__(self, lhs, rhs)
        self.order_type = order_type
        self.comp = comp

    def __str__(self):
        return str(self.lhs) + " ~> " + str(self.rhs) + str(self.order_type) + str(self.comp)


class UCC(DEP):
    def __str__(self):
        return str(self.lhs)
