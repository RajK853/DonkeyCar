from . import donkey_net


def get_donkey_net_func(net_name):
    donkey_net_func = getattr(donkey_net, net_name)
    return donkey_net_func
