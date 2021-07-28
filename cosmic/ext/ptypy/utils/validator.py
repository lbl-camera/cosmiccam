# -*- coding: utf-8 -*-
"""\
Parameter validation. This module parses the file
``resources/parameters_descriptions.csv`` to extract the parameter
defaults for |ptypy|. It saves all parameters in the form of 
a :py:class:`PDesc` object, which are flat listed in 
`parameter_descriptions` or in `entry_points_dct`, which only contains
parameters with subparameters (children).

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import ast
import weakref
from collections import OrderedDict


#! Validator message codes
CODES = dict(
    PASS=1,
    FAIL=0,
    UNKNOWN=2,
    MISSING=3,
    INVALID=4)

#! Inverse message codes
CODE_LABEL = dict((v, k) for k, v in CODES.items())

"""
# Populate the dictionary of all entry points.

# All ptypy parameters in an ordered dictionary as (entry_point, PDesc) pairs.
parameter_descriptions = OrderedDict()
del OrderedDict

# All parameter containers in a dictionary as (entry_point, PDesc) pairs.
# Subset of :py:data:`parameter_descriptions`
entry_points_dct = {}
"""

# Logging levels
import logging

_logging_levels = dict(
    PASS=logging.INFO,
    FAIL=logging.CRITICAL,
    UNKNOWN=logging.WARN,
    MISSING=logging.WARN,
    INVALID=logging.ERROR)

del logging

_typemap = {'int': 'int',
           'float': 'float',
           'complex': 'complex',
           'str': 'str',
           'bool': 'bool',
           'tuple': 'tuple',
           'list': 'list',
           'array': 'ndarray',
           'Param': 'Param',
           'None': 'NoneType',
           'file': 'str',
           '': 'NoneType'}

_evaltypes = ['int','float','tuple','list','complex']
_copytypes = ['str','file']

class Parameter(object):
    """
    """
    def __init__(self, parent=None, 
                       name = 'any',
                       separator='.', 
                       info=None):
                
        #: Name of parameter
        self.name = name
        
        #: Parent parameter (:py:class:`Parameter` type) if it has one.
        self.parent = parent
        
        self.descendants = {}
        """ Flat list of all sub-Parameters. These are weak references
        if not root."""
        
        #: Hierarchical tree of sub-Parameters.
        self.children = {}
        
        self.separator = separator
        
        self.required = [] 
        self.optional = []
        self.info = {}
        self._parse_info(info)
        
        if self._is_child:
            import weakref
            self.descendants = weakref.WeakValueDictionary()
            

        self.num_id = 0
        self.options = dict.fromkeys(self.required,'')
        self._all_options = {}
        
    @property
    def descendants_options(self):
        return list(self._all_options.keys())
        
    @property
    def _is_child(self):
        """
        Type check
        """
        return type(self.parent) is self.__class__

        
    def _parse_info(self,info=None):
        if info is not None:
            self.info.update(info)
            
            r = []
            o = []
        
            for option,text in self.info.items():
                if ('required' in text or 'mandatory' in text):
                    r+=[option]
                else:
                    o+=[option]
            self.required = r
            self.optional = o

            
    def _new(self, name=None):
        n = name if name is not None and str(name)==name else 'ch%02d' % len(self.descendants)
        return self.__class__(parent = self, 
                                 separator = self.separator,
                                 info = self.info,
                                 name =n
                                 )
        
    def _name_descendants(self, separator =None):
        """
        This function transforms the flat list of descendants
        into tree hierarchy. Creates roots if paramater has a 
        dangling root.
        """
        sep = separator if separator is not None else self.separator
               
        for name, desc in self.descendants.items():
            if sep not in name:
                desc.name = name
                self.children[name] = desc
            else:
                names = name.split(sep)
                
                nm = names[0]
                
                p = self.descendants.get(nm)
                
                if p is None:
                    # Found dangling parameter. Create a root
                    p = self._new(name = nm)
                    self._new_desc(nm, p)
                    self.children[nm] = p
                    
                # transfer ownership
                p.descendants[sep.join(names[1:])] = desc
                desc.parent = p
        
        # recursion
        for desc in self.children.values():
            desc._name_descendants()
            
    def _get_root(self):
        """
        Return root of parameter tree.
        """
        if self.parent is None:
            return self
        else:
            return self.parent._get_root()
    
    def _get_path(self):
        """
        Return root of parameter tree.
        """
        if self.parent is None:
            return self
        else:
            return self.parent._get_root()
            
    def _store_options(self,dct):
        """
        Read and store options and check that the the minimum selections
        of options is present.
        """
        
        if self.required is not None and type(self.required) is list:
            missing = [r for r in self.required if r not in list(dct.keys())]
            if missing:
                raise ValueError('Missing required option(s) <%s> for parameter %s.' % (', '.join(missing),self.name))

        self.options = dict.fromkeys(self.required)
        self.options.update(dct)
        
    @property    
    def root(self):
        return self._get_root()
            
    @property
    def path(self):
        if self.parent is None:
            return self.name
        else:
            return self.parent.path + self.separator + self.name
            
    def _new_desc(self, name, desc, update_in_parent = True):
        """
        Update the new entry to the root.
        """
        self.descendants[name] = desc
        
        # add all options to parent class
        self._all_options.update(desc.options)
        
        if update_in_parent:
            if self._is_child:
                # You are not the root
                self.parent._new_desc(self.name+self.separator+name,desc)
            else:
                # You are the root. Do root things here.
                pass
                
    def load_csv(self, fbuffer, **kwargs):
        """
        Load from csv as a fielded array. Keyword arguments are passed
        on to csv.DictReader
        """
        from csv import DictReader
        CD = DictReader(fbuffer, **kwargs)
        
        if 'level' in CD.fieldnames:
            chain = []
            
            # old style CSV, name + level sets the path
            for num, dct in enumerate(list(CD)):
            
                # Get parameter name and level in the hierarchy
                level = int(dct.pop('level'))
                name = dct.pop('name')
            
                # translations
                dct['help']= dct.pop('shortdoc')
                dct['doc']= dct.pop('longdoc')
                if dct.pop('static').lower()!='yes': continue
            
                desc = self._new(name)
                desc._store_options(dct)
                desc.num_id = num
                
                if level == 0:  
                    chain = [name]
                else:
                    chain = chain[:level]+[name]
            
                self._new_desc(self.separator.join(chain), desc)
        else:
            # new style csv, name and path are synonymous
            for dct in list(CD):
                name = dct['path']
                desc = self._new(name)
                desc._store_options(dct)
                self._new_desc(name, desc)
        
        self._name_descendants()
        
    def save_csv(self, fbuffer, **kwargs):
        """
        Save to fbuffer. Keyword arguments are passed
        on to csv.DictWriter
        """
        from csv import DictWriter
        
        fieldnames = self.required + self.optional
        fieldnames += [k for k in self._all_options.keys() if k not in fieldnames]
        
        DW = DictWriter(fbuffer, ['path'] + fieldnames)
        DW.writeheader()
        for key in sorted(self.descendants.keys()):
            dct = {'path':key}
            dct.update(self.descendants[key].options)
            DW.writerow(dct)
        
    def load_json(self,fbuffer):
        
        raise NotImplementedError
    
    def save_json(self,fbuffer):
        
        raise NotImplementedError
    
    
    def load_conf_parser(self,fbuffer, **kwargs):
        """
        Load Parameter defaults using Pythons ConfigParser
        
        Each parameter each parameter occupies its own section. 
        Separator characters in sections names map to a tree-hierarchy.
        
        Keyword arguments are forwarded to `ConfigParser.RawConfigParser`
        """
        from configparser import RawConfigParser as Parser
        parser = Parser(**kwargs)
        parser.readfp(fbuffer)
        parser = parser
        for num, sec in enumerate(parser.sections()):
            desc = self._new(name=sec)
            desc._store_options(dict(parser.items(sec)))
            self._new_desc(sec, desc)
        
        self._name_descendants()
        return parser
            
    def save_conf_parser(self,fbuffer, print_optional=True):
        """
        Save Parameter defaults using Pythons ConfigParser
        
        Each parameter each parameter occupies its own section. 
        Separator characters in sections names map to a tree-hierarchy
        """
        from configparser import RawConfigParser as Parser
        parser = Parser()
        dct = self.descendants
        for name in sorted(dct.keys()):
            if dct[name] is None:
                continue
            else:
                parser.add_section(name)
                for k,v in self.descendants[name].options.items():
                    if (v or print_optional) or (k in self.required):
                        parser.set(name, k, v)
        
        parser.write(fbuffer)
        return parser
        
    def make_doc_rst(self,prst, use_root=True):
        
        Header=  '.. _parameters:\n\n'
        Header+= '************************\n'
        Header+= 'Parameter tree structure\n'
        Header+= '************************\n\n'
        prst.write(Header)
        
        root = self.get_root() # if use_root else self
        shortdoc = 'shortdoc'
        longdoc = 'longdoc'
        default = 'default'
        lowlim ='lowlim'
        uplim='uplim'
        
        start = self.get_root()
        
        for name,desc in root.descendants.items():
            if name=='':
                continue
            if hasattr(desc,'children') and desc.parent is root:
                prst.write('\n'+name+'\n')
                prst.write('='*len(name)+'\n\n')
            if hasattr(desc,'children') and desc.parent.parent is root:
                prst.write('\n'+name+'\n')
                prst.write('-'*len(name)+'\n\n')
            
            opt = desc.options

            prst.write('.. py:data:: '+name)
            #prst.write('('+', '.join([t for t in opt['type']])+')')
            prst.write('('+opt['type']+')')
            prst.write('\n\n')
            num = str(opt.get('ID'))
            prst.write('   *('+num+')* '+opt[shortdoc]+'\n\n')
            prst.write('   '+opt[longdoc].replace('\n','\n   ')+'\n\n')
            prst.write('   *default* = ``'+str(opt[default]))
            if opt[lowlim] is not None and opt[uplim] is not None:
                prst.write(' (>'+str(opt[lowlim])+', <'+str(opt[uplim])+')``\n')
            elif opt[lowlim] is not None and opt[uplim] is None:
                prst.write(' (>'+str(opt[lowlim])+')``\n')
            elif opt[lowlim] is None and opt[uplim] is not None:
                prst.write(' (<'+str(opt[uplim])+')``\n')
            else:
                prst.write('``\n')
                
            prst.write('\n')
        prst.close()

class ArgParseParameter(Parameter):
    DEFAULTS = OrderedDict(
        default = 'Default value for parameter.',
        help = 'A small docstring for command line parsing (required).',
        choices = 'If parameter is list of choices, these are listed here.'
    )
    def __init__(self, *args,**kwargs):
        
        info = self.DEFAULTS.copy()
        ninfo = kwargs.get('info')
        if ninfo is not None:
            info.update(ninfo)
            
        kwargs['info'] = info
        
        super(ArgParseParameter, self).__init__(*args,**kwargs)

    @property
    def help(self):
        """
        Short descriptive explanation of parameter
        """
        return self.options.get('help', '')

    @property
    def literal_default(self):
        """
        Returns default as raw string or empty string
        """
        return str(self.options.get('default', ''))
        
    @property
    def default(self):
        """
        Returns default as a Python type
        """
        default = self.literal_default
        
        if not default:
            return None
        else:
            return self.eval(default)


    def eval(self,val):
        """
        A more verbose wrapper around `ast.literal_eval`
        """
        try:
            return ast.literal_eval(val)
        except ValueError as e:
            msg = e.message+". could not read %s for parameter %s" % (val,self.name)
            raise ValueError(msg)
            
    @property
    def choices(self):
        """
        If parameter is a list of choices, these are listed here.
        """
        # choices is an evaluable list
        c =  self.options.get('choices', '')
        #print c, self.name
        if str(c)=='':
            c=None
        else:
            try:
                c = ast.literal_eval(c.strip())
            except SyntaxError('Evaluating `choices` %s for parameter %s failed' %(str(c),self.name)):
                c = None
        
        return c
    

    def make_default(self, depth=1):
        """
        Creates a default parameter structure, from the loaded parameter
        descriptions in this module
        
        Parameters
        ----------            
        depth : int
            The depth in the structure to which all sub nodes are expanded
            All nodes beyond depth will be returned as empty dictionaries
            
        Returns
        -------
        pars : dict
            A parameter branch as nested dicts.
        
        Examples
        --------
        >>> from ptypy import parameter
        >>> print parameter.children['io'].make_default()
        """
        out = {}
        if depth<=0:
            return out
        for name,child in self.children.items():
            if child.children and child.default is None:
                out[name] = child.make_default(depth=depth-1)
            else:
                out[name] = child.default
        return out
        
    def _get_type_argparse(self):
        """
        Returns type or callable that the argparser uses for 
        reading in cmd line argements.
        """
        return type(self.default)
        
    def add2argparser(self,parser=None, prefix='',
                        excludes=('scans','engines'), mode = 'add'):
        
        sep = self.separator
        
        pd = self
               
        argsep='-'

        if parser is None:
            from argparse import ArgumentParser
            description = """
            Parser for %s
            Doc: %s
            
            Please be aware that Python string quotations are escaped in Bash.
            """ % (pd.name, pd.help)
            parser = ArgumentParser(description=description)
        
        # overload the parser
        if not hasattr(parser,'_aux_translator'): 
            parser._aux_translator={}
        
        
        # get list of descendants and remove separator
        ndesc = dict((k.replace(sep,argsep),v) for k,v in self.descendants.items())
        
        
        groups = {}
        
        for name, pd in ndesc.items():
            if pd.name in excludes: continue
            if pd.children:
                groups[name] = parser.add_argument_group(title=prefix+name, description=pd.help)

        for name, pd in ndesc.items():
            
            if pd.name in excludes: continue
            up = argsep.join(name.split(argsep)[:-1])
            # recursive part
            parse = groups.get(up,parser)

            """
            # this should be part of PDesc I guess.
            typ = type(pd.default)
            
            for t in pd.type:
                try:
                    typ= eval(t)
                except BaseException:
                    continue
                if typ is not None:
                    break

            if typ is None:
                u.verbose.logger.debug('Failed evaluate type strings %s of parameter %s in python' % (str(pd.type),name))
                return parser
                
            if type(typ) is not type:
                u.verbose.logger.debug('Type %s of parameter %s is not python type' % (str(typ),name))
                return parser
            """
            typ = pd._get_type_argparse()
            
            if typ is bool:
                flag = '--no-'+name if pd.default else '--'+name
                action='store_false' if pd.default else 'store_true'
                parse.add_argument(flag, dest=name, action=action, 
                                 help=pd.help )
            else:
                d = pd.literal_default
                defstr =  d.replace('%(','%%(') if str(d)==d else str(d)
                parse.add_argument('--'+name, dest=name,default = d, choices=pd.choices, 
                                 help=pd.help +' (default=%s)' % defstr)            
        
            parser._aux_translator[name] = pd
            
        return parser
    
    def _from_argparser(self,args,parser):
        for k,pd in parser._aux_translator.items():
            pd.options['default'] = args[k]
            
    def parse_args(self,*args,**kwargs):
        
        parser= self.add2argparser(**kwargs)
        args = parser.parse_args(*args).__dict__
        self._from_argparser(args,parser)
        
class EvalParameter(ArgParseParameter):
    """
    Small class to store all attributes of a ptypy parameter
    
    """
    _typemap = {'int': 'int',
           'float': 'float',
           'complex': 'complex',
           'str': 'str',
           'bool': 'bool',
           'tuple': 'tuple',
           'list': 'list',
           'array': 'ndarray',
           'Param': 'Param',
           'None': 'NoneType',
           'file': 'str',
           '': 'NoneType'}

    _evaltypes = ['int','float','tuple','list','complex']
    _copytypes = ['str','file']
    
    DEFAULTS = OrderedDict(
        default = 'Default value for parameter (required).',
        help = 'A small docstring for command line parsing (required).',
        type = 'Komma separated list of acceptable types.',
        doc = 'A longer explanation for the online docs.',
        uplim = 'Upper limit for scalar / integer values',
        lowlim = 'Lower limit for scalar / integer values',
        choices = 'If parameter is list of choices, these are listed here.',
        userlevel = """User level, a higher level means a parameter that is 
                    less likely to vary or harder to understand.""",
    )
     
    def __init__(self, *args,**kwargs):

        kwargs['info']=self.DEFAULTS.copy()
        
        super(EvalParameter, self).__init__(*args,**kwargs)
        
    @property
    def default(self):
        """
        Returns default as a Python type
        """
        default = str(self.options.get('default', ''))
        
        # this destroys empty strings
        default = default if default else None
        
        if default is None:
            out = None
        # should be only strings now
        elif default.lower()=='none':
            out = None
        elif default.lower()=='true':
            out = True
        elif default.lower()=='false':
            out = False
        elif self.is_evaluable:
            out = ast.literal_eval(default)
        else:
            out = default
        
        return out 
        
    @property
    def type(self):
            
        types = self.options.get('type', None)
        tm = self._typemap
        if types is not None:
            types = [tm[x.strip()] if x.strip() in tm else x.strip() for x in types.split(',')]
        
        return types        
       
    @property
    def limits(self):
        if self.type is None:
            return None, None
            
        ll = self.options.get('lowlim', None)
        ul = self.options.get('uplim', None)
        if 'int' in self.type:
            lowlim = int(ll) if ll else None
            uplim = int(ul) if ul else None
        else:
            lowlim = float(ll) if ll else None
            uplim = float(ul) if ul else None
            
        return lowlim,uplim
        
    @property
    def doc(self):
        """
        Longer documentation, may contain *sphinx* inline markup.
        """
        return self.options.get('doc', '')

    @property
    def userlevel(self):
        """
        User level, a higher level means a parameter that is less 
        likely to vary or harder to understand.
        """
        # User level (for gui stuff) is an int
        ul = self.options.get('userlevel', 1)
        
        return int(ul) if ul else None
     
    @property
    def is_evaluable(self):
        for t in self.type:
            if t in self._evaltypes:
                return True
                break
        return False
        
    
    def check(self, pars, walk):
        """
        Check that input parameter pars is consistent with parameter description.
        If walk is True and pars is a Param object, checks are also conducted for all
        sub-parameters.
        """
        ep = self.path
        out = {}
        val = {}

        # 1. Data type
        if self.type is None:
            # Unconclusive
            val['type'] = CODES['UNKNOWN']
            val['lowlim'] = CODES['UNKNOWN']
            val['uplim'] = CODES['UNKNOWN']
            return {ep : val}
        else:
            val['type'] = CODES['PASS'] if (type(pars).__name__ in self.type) else CODES['FAIL']

        # 2. limits
        lowlim, uplim = self.limits
        
        if lowlim is None:
            val['lowlim'] = CODES['UNKNOWN']
        else:
            val['lowlim'] = CODES['PASS'] if (pars >= self.lowlim) else CODES['FAIL']
        if uplim is None:
            val['uplim'] = CODES['UNKNOWN']
        else:
            val['uplim'] = CODES['PASS'] if (pars <= self.uplim) else CODES['FAIL']

        # 3. Extra work for parameter entries
        if 'Param' in self.type:
            # Check for missing entries
            for k, v in self.children.items():
                if k not in pars:
                    val[k] = CODES['MISSING']

            # Check for excess entries
            for k, v in pars.items():
                if k not in self.children:
                    val[k] = CODES['INVALID']
                elif walk:
                    # Validate child
                    out.update(self.children[k].check(v, walk))

        out[ep] = val
        return out
        
    def validate(self,pars, walk=True, raisecodes=[CODES['FAIL'], CODES['INVALID']]):
        """
        Check that the parameter structure `pars` matches the documented 
        constraints for this node / parameter.
    
        The function raises a RuntimeError if one of the code in the list 
        `raisecodes` has been found. If raisecode is empty, the function will 
        always return successfully but problems will be logged using logger.
    
        Parameters
        ----------
        pars : Param
            A parameter set to validate
        
        walk : bool
            If ``True`` (*default*), navigate sub-parameters.
        
        raisecodes: list
            List of codes that will raise a RuntimeError.
        """
        from ptypy.utils.verbose import logger
        
        d = self.check(pars, walk=walk)
        do_raise = False
        for ep, v in d.items():
            for tocheck, outcome in list(v.items()):
                logger.log(_logging_levels[CODE_LABEL[outcome]], '%-50s %-20s %7s' % (ep, tocheck, CODE_LABEL[outcome]))
                do_raise |= (outcome in raisecodes)
        if do_raise:
            raise RuntimeError('Parameter validation failed.')
            
    def sanity_check(self, depth=10):
        """
        Checks if default parameters from configuration are 
        self-constistent with limits and choices.
        """
        self.validate(self.make_default(depth =depth))


    
