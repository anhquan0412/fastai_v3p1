from exp.nb_07 import *

def get_batch(dl, run):
    run.xb,run.yb = next(iter(dl))
    for cb in run.cbs: cb.set_runner(run)
    run('begin_batch')
    return run.xb,run.yb

def find_modules(m, cond):
    if cond(m): return [m]
    return sum([find_modules(o,cond) for o in m.children()], [])

def is_lin_layer(l):
    lin_layers = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)
    return isinstance(l, lin_layers)

def lsuv_module(m, xb):
    h = Hook(m, append_mean)

    if getattr(m, 'bias', None) is not None:
        while mdl(xb) is not None and abs(h.mean) > 1e-3:
            m.bias.data -= h.mean

    while mdl(xb) is not None and abs(h.std-1) > 1e-3:
        m.weight.data /= h.std

    h.remove()
    return h.mean,h.std