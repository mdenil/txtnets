__author__ = 'mdenil'


import numpy as np
import unittest

from cpu.optimize import sgd
from cpu.optimize import update_rule

class NarrowQuadraticObjective(object):
    def __init__(self):
        self.A = np.array([[10.0, 0.99],[0.99, 10.0]])

    def evaluate(self, model):
        cost = np.dot(np.dot(model.w.T, self.A), model.w)
        grads = [2*np.dot(self.A, model.w)]
        return float(cost), grads


class PointModel(object):
    def __init__(self, w):
        self.w = np.asarray(w).reshape((-1,1)).astype(np.float)

    def params(self):
        return [self.w]



class UpdateRuleSkeleton(object):
    def setUp(self):
        self.objective = NarrowQuadraticObjective()
        self.model = PointModel([10.0, 7.0])

    def test_update_rule(self):
        optimizer = sgd.SGD(model=self.model, objective=self.objective, update_rule=self.get_update_rule())

        for count, info in enumerate(optimizer):
            if count > 100:
                break

        print self.model.w.T
        assert np.dot(self.model.w.T, self.model.w) < 1e-2

class TestBasicUpdateRule(UpdateRuleSkeleton, unittest.TestCase):
    def get_update_rule(self):
        return update_rule.Basic(learning_rate=1e-2)

class TestAdaDeltaUpdateRule(UpdateRuleSkeleton, unittest.TestCase):
    def get_update_rule(self):
        return update_rule.AdaDelta(
            rho=0.9,
            epsilon=1e-2,
            model_template=self.model)


class TestAdaGradUpdateRule(UpdateRuleSkeleton, unittest.TestCase):
    def get_update_rule(self):
        return update_rule.AdaGrad(
            gamma=3.0,
            model_template=self.model)


class TestMomentumUpdateRule(UpdateRuleSkeleton, unittest.TestCase):
    def get_update_rule(self):
        return update_rule.Momentum(
            momentum=0.9,
            epsilon=1e-2,
            model_template=self.model)

class TestNAGUpdateRule(UpdateRuleSkeleton, unittest.TestCase):
    def get_update_rule(self):
        return update_rule.NesterovAcceleratedGradient(
            momentum=0.9,
            epsilon=1e-2,
            model_template=self.model)




if __name__ == "__main__":
    import matplotlib.pyplot as plt

    objective = NarrowQuadraticObjective()

    model = PointModel([10.0, 10.0])


    update_rules = [
        update_rule.Basic(learning_rate=1e-2),
        update_rule.AdaDelta(
            rho=0.2,
            epsilon=1e-2,
            model_template=model),
        update_rule.AdaGrad(
            rho=0.9,
            eta=0.5,
            epsilon=1e-2,
            model_template=model),
        update_rule.Momentum(
            momentum=0.9,
            epsilon=1e-2,
            model_template=model),
        update_rule.NesterovAcceleratedGradient(
            momentum=0.9,
            epsilon=1e-2,
            model_template=model),
        ]

    costs = []
    for update_rule in update_rules:
        model = PointModel([10.0, 7.0])
        optimizer = sgd.SGD(model=model, objective=objective, update_rule=update_rule)

        rule_costs = []

        for count, info in enumerate(optimizer):
            w_model = info['model'].w
            # print info['cost'],  np.dot(w_model.T, w_true) / ( np.sqrt(np.dot(w_true.T, w_true)) * np.sqrt(np.dot(w_model.T, w_model)) )
            # print info['model'].w

            # rule_costs.append(info['cost'])
            # print info['model'].w.T

            rule_costs.append(float(np.dot(info['model'].w.T, info['model'].w)))
            if count > 100:
                break

        costs.append(rule_costs)


    fig = plt.figure()
    ax = fig.gca()
    for update_rule, rule_costs in zip(update_rules, costs):
        ax.plot(rule_costs, label=str(update_rule))
    plt.legend()
    plt.show()