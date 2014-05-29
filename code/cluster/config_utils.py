__author__ = 'davidm, mdenil'

import itertools


def flatten(l):
    """
    Turn a list of list of (list of ...) dicts into a list of dicts.
    """
    result = []
    for x in l:
        if type(x) is list:
            result.extend(flatten(x))
        else:
            result.append(x)
    return result


def expand(d):
    """
    Turn a dict of lists into a list of dicts by taking the cross product over all of the list elements.
    """
    # make sure everything is a list
    for k, v in d.iteritems():
        if type(v) is not list:
            d[k] = [v]

    # take cross product
    product = [x for x in apply(itertools.product, d.values())]
    return flatten([dict(zip(d.keys(), p)) for p in product])


def product(l):
    """
    Takes a sequence of lists of dictionaries and performs the algebraic product:

    (a + b) * (c) * (d + e) => (a * c * d) + (a * c * e) + (b * c * d) + (b * c * e)

    Addition is concatenation of configuration lists: a + b means configuration a and configuration b
    Multiplication is a cross product of the keys in two configurations.
    """
    result = []
    join_tuples = [join_tuple for join_tuple in itertools.product(*l)]
    for join in join_tuples:
        d = flatten([sub.items() for sub in join])
        result.append(dict(d))
    return result


def remove_from_dict(config, params):
    result = config.copy()
    for p in params:
        if p in result:
            del result[p]

    return result


def shorten(param_name):
    if not isinstance(param_name, basestring) or param_name.find('_') == -1:
        return param_name
    else:
        shortened = ''
        next = 0
        while True:
            shortened += param_name[next]
            next = param_name.find('_', next) + 1
            if next == 0:
                break

        return shortened


def get_config_string(config):
    config_strings = []
    for k, v in sorted(config.items()):
        config_strings.append('%s~%s' % (shorten(k), shorten(v)))

    return '-'.join(config_strings)
