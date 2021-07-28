__all__ = ['addr2ipport', 'Adict', 'eval_hdr', 'ScanInfo']

from collections import OrderedDict


class ScanInfo(object):
    _translation = {
        'num_pixels_x': 'exp_num_x',
        'num_pixels_y': 'exp_num_y',
        'step_size_x': 'exp_step_x',
        'step_size_y': 'exp_step_y',
        'background_pixels_x': 'dark_num_x',
        'background_pixels_y': 'dark_num_y',
        'path1': 'dark_dir',
        'path2': 'exp_dir',
    }

    def __init__(self):
        pass

    def read_file(self, fname):
        with open(fname, 'r') as f:
            din = dict([l.strip().split() for l in f.readlines()])
            f.close()
        self._scan_info_raw = din
        return self._convert(din)

    def read_tcp(self, tcpstr):
        infos = tcpstr.split(',')
        din = {'hdr_path': infos.pop(0).strip()}
        din.update([kv.strip().split() for kv in infos])
        self._scan_info_raw = din
        return self._convert(din)

    def _convert(self, din):
        import ast
        dt = {}
        for k, v in din.items():
            try:
                vi = ast.literal_eval(v)
            except:
                vi = v
            dt[self._translation.get(k, k)] = vi

        d = OrderedDict([(k, dt[k]) for k in sorted(dt.keys())])
        try:
            Ny = d['exp_num_y']
            Nx = d['exp_num_x']
            d['dark_num_total'] = d['dark_num_x'] * d['dark_num_y']
            d['double_exposure'] = True if d['isDoubleExp'] == 1 else False
            d['exp_num_total'] = Nx * Ny
            dx, dy = d['exp_step_x'], d['exp_step_y']
            d['translations'] = [(y * dy, x * dx) for y in range(Ny) for x in range(Nx)]

        except KeyError as e:
            warnings.warn('Translation extraction failed: (%s)' % e.message)  # create postiion
            d['dark_num_total'] = None
            d['exp_num_total'] = None
            d['translations'] = None

        self._scan_info = d

        return d

    def to_json(self, dct=None, **kwargs):
        import json
        dct = self._scan_info if dct is None else dct
        dct.update(kwargs)
        return json.dumps(dct)


def addr2ipport(addr, ip='localhost', port='8880'):
    """
    Translates address string `addr` of the form "%s:%d" to ip and port.
    """
    import urllib.request, urllib.error, urllib.parse
    addr = addr.strip().split(':')
    if len(addr) == 1:
        try:
            return ip, int(addr[0])
        except TypeError:
            return str(addr[0]), port
    elif len(addr) == 2:
        return str(addr[0]), int(addr[1])
    else:
        raise ValueError('Address string %s not interpretable' % addr)


class Adict(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def _to_dict(self):
        res = {}
        for k, v in self.__dict__.items():
            if v.__class__ is self.__class__:
                res[k] = v._to_dict()
            else:
                res[k] = v
        return res


def eval_hdr(fname, as_attributes=False):
    """
    Reads .hdr file and uses string replacement and eval to load
    """
    s = 'Adict(' if as_attributes else 'dict('
    f = open(fname)
    t = f.read().replace('\r\n', '')
    f.close()
    t = t.replace('{', s).replace('}', ')')
    t = t.replace(';', ',').replace('false', 'False').replace('true', 'True')
    d = eval(s + t + ')')
    return d


def readASCIIMatrix(filename, separator='\t'):
    data = []
    f = open(filename, 'r')
    for line in f:
        row = line.split(separator)
        data.append(row[0:len(row) - 1])
    return np.array(data).astype('float')
