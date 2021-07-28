import os
import glob
import time
import numpy as np
from scipy.misc import toimage
from matplotlib import cm as cmaps
import h5py

import sqlite3
from collections import OrderedDict
from collections import namedtuple


example_cxi = '/home/benders/phasis/groups/cosmic/Data/2017/06/170615/170615041/002/NS_170615041.cxi'
#example_cxi = '/global/groups/cosmic/Data/2017/06/170615/170615041/002/NS_170615041.cxi'

def get_image_data_from_cxi(fname):

    with h5py.File(fname,'r') as f:
        
        entries = sorted([e for e in list(f['entry_1']) if e.startswith('image')])
        if not entries:
            res = None
        else:
            res = f['entry_1/' + entries[-1] + '/data'].value
        
        f.close()
        
    return res
    """
    print entries
    if 'image_latest' in entries:
        image_offset = 1
    else: 
        image_offset = 0
    try: 
        n = max(loc for loc, val in enumerate(entryList) if not(val.rfind('image'))) - image_offset
    except:
        return None
    else:
        print "Found %d images" % n
        image = []
        for i in range(1,n + 1): 
            print "loading image: %s" %(i)
            image.append(f['entry_1/image_' + str(i) + '/data'].value)
        return image
    """
    
class Adict(object):
    
    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)
        
    def _to_dict(self):
        res = {}
        for k,v in self.__dict__.items():
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
    t = f.read().replace('\r\n','')
    f.close()
    t = t.replace('{',s).replace('}',')')
    t = t.replace(';',',').replace('false','False').replace('true','True')
    d = eval(s+t+')')
    return d
    
def readASCIIMatrix(filename,separator='\t'):

    data = []
    f = open(filename,'r')
    for line in f:
        row = line.split(separator)
        data.append(row[0:len(row)-1])
    return np.array(data).astype('float')

def _subsec(name):
    return name+'\n' + len(name) * '^' +'\n\n'


def _pic(fname, height = 100):
    return '  .. image:: ' + picname +'\n'+' '*5+':scale: 200\n'+ ' '*5+':height: %d\n\n' % height
     
     

class DBobject(object):
    
    _DB = None 
    
    _DB_UNIQUE_KEY = None
    
    RECORDS = OrderedDict(
        haus='garten',
        zahl=1,
        other =0
    )
    
    def __init__(self,db_table_name='', **kwargs):
        
        self._db_records = self.RECORDS.copy()
        self._db_tn = db_table_name if db_table_name else self.__class__.__name__
        self._update_records(kwargs)
        c = self._DB.cursor()
        
        # check if table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?",(self._db_tn,))

        if c.fetchone() is None:
            cols = list(self.RECORDS.keys())
            execstr = "CREATE TABLE %s " % self._db_tn
            execstr += "(%s)" % ','.join(cols)
            c.execute(execstr)
            
        self._DB.commit()
        
        rowid = self._db_find_rowid()
        
        if not rowid:
            # No row entry yet in DB, make an entry
            self._rowid = self._db_insert()
        elif len(rowid)>1:
            # Entry is non-unique -- This should not happen
            raise LookupError('Double entry in database.')
        else:
            self._rowid = int(rowid[0])
               
        
    def _update_records(self,dct):
        r = self._db_records
        for k,v in dct.items():
            if k in list(r.keys()):
                r[k] = v
    
    def _db_find_rowid(self):
        return [row[0][1] for row in self._db_select(None,['rowid'])]
        
    def _db_select(self, match = None, select = None):
        """
        Returns coulumns `select` for rows having the same value as self
        in columns `match`. None translates to all columns.
        """
        if match is None:
            match = list(self._db_records.keys())
        if select is None:
            select =  list(self._db_records.keys())
        
        c = self._DB.cursor()
        execstr = "SELECT %s " % ', '.join(select)
        execstr += "FROM %s WHERE " % self._db_tn
        execstr += " AND ".join(['%s=?' %k for k in match])
        args = [self._db_records[k] for k in match]
        print(execstr,args)
        c.execute(execstr,tuple(args))
        res = [zip(select,r) for r in c.fetchall()]
        return res
        
    def _db_insert(self,dct=None):
        execstr = "INSERT INTO %s " % self._db_tn
        c = self._DB.cursor()
        c.execute(execstr + "VALUES (" + ",".join(['?']*len(list(self.RECORDS.keys()))) + ")", list(self._db_records.values()))
        self._DB.commit()
        return c.lastrowid
        
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

class PtychoHdr(Base):
    """
    RECORDS = OrderedDict()
    RECORDS.update([
    ('name','INTEGER'),
    ('year','INTEGER'),
    ('month','INTEGER'),
    ('day', 'INTEGER'),
    ('scan','INTEGER'),
    ('date','TEXT'),
    ('hdr', 'TEXT'),
    ('cxipath', 'TEXT'),
    ('area', 'REAL'),
    ('points', 'REAl'),
    ('step', 'REAL'),
    ('angle', 'REAL'),
    ('annotion', 'TEXT'),
    ('reconstructed', 'INTEGER'),
    ])
    """
    __tablename__ = 'ptycho'
    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String)
    datetime = sa.Column(sa.DateTime)
    scan = sa.Column(sa.Integer)
    cxipath = sa.Column(sa.String)
    imgpath = sa.Column(sa.String)
    reconstructed = sa.Column(sa.Integer)
    area = sa.Column(sa.Float)
    width = sa.Column(sa.Float)
    height = sa.Column(sa.Float)
    points = sa.Column(sa.Integer)
    step = sa.Column(sa.Float)
    angle = sa.Column(sa.Float)
    annotation = sa.Column(sa.Text)
    hdr = sa.Column(sa.Text)
    energy = sa.Column(sa.Float)
    estack = sa.Column(sa.Integer)
    
    cpal = np.array([cmaps.bone(i) for i in range(256)]) * 255 
    
    def extract_from_hdr(self, hdr):
        
        try:
            dhdr = eval_hdr(hdr)
        except SyntaxError:
            return False
            
        scan_def = dhdr.get('ScanDefinition')
        if scan_def is None:
            print(os.path.split(hdr)[1], 'Could not find ScanDefinition')
            return False
        else:
            print(os.path.split(hdr)[1],scan_def['Type'])
        
        if scan_def['Type']!='Ptycho Image Scan':
            return False
        
        from datetime import datetime
        tm = datetime.strptime(dhdr['Time'],'%Y %B %d %H:%M:%S')
        self.datetime = tm
        
        #out['date'] = time.strftime('%a %b %d %H:%M:%S %Y',tm)
        #out['year'] = int(tm.tm_year) 
        #out['month'] = int(tm.tm_mon) 
        #out['day'] = int(tm.tm_mday) 
        
        # Could be property
        base,fname = os.path.split(hdr)
        name = fname.replace('.hdr','')
        self.id = int(name[3:])
        self.scan = int(name[-3:])
        self.name = name
        # out['name'] = name
        self.angle = dhdr['AUX11']['LastPosition']

        spat_regions = dhdr['ImageScan']['SpatialRegions']
        num_reg = spat_regions[0]
        npoints = 0
        for ii in range(1,num_reg+1):
            pars = spat_regions[ii]
            n = pars['YPoints'] * pars['XPoints']
            if n >= npoints:
                # this is assumed to be the actual ptycho scan
                self.area = pars['XRange'] * pars['YRange']
                self.width = pars['XRange']
                self.height = pars['YRange']
                self.step = 0.5 * (pars['XStep'] + pars['YStep'])
                npoints = n
                self.points = n
            else:
                break
        
        #energy_reg, pars = dhdr['ImageScan']['EnergyRegions']
        self.energy = np.mean(scan_def['StackAxis']['Points'][1:])
        self.estack = scan_def['StackAxis']['Points'][0]
        #cxi        
        cxi_search = base + '/' + name[3:] + '/%03d/' + fname.replace('.hdr','.cxi')
        for i in range(6):
            fcxi = cxi_search % i
            if os.path.exists(fcxi):
                self.cxipath = fcxi
                with h5py.File(fcxi,'r') as f:
                    entries = sorted([e for e in list(f['entry_1']) if e.startswith('image')])
                    self.reconstructed = len(entries) - int('image_latest' in entries)
                    f.close()
                break
       
        return True
        #energy_reg, pars = dhdr['ImageScan']['EnergyRegions']
    
    def png_from_cxi(self, img_path = None, rerender=False, caption = None):
        
        if self.cxipath is None:
            return None
        else:
            data = get_image_data_from_cxi(self.cxipath)
            if data is not None and img_path is not None:
                if not os.path.exists(img_path):
                    os.makedirs(img_path)
                    
                pfile = img_path + self.name + '.png'
                self.imgpath = pfile
                if not os.path.exists(pfile) or rerender:
                    A = np.abs(data)
                    img = toimage(A, pal=self.cpal[:,:3], mode='P')
                    pfile = img_path + self.name + '.png'
                    img.save(pfile)
                return pfile
            else:
                return None
        
                """
                with open(pfile.replace('.png','.txt'),'w') as f:
                    f.write(write_caption(out))
                    f.close()
                break
                """

    def __repr__(self):
        return "<PtychoHdr(name='%s')>" % self.name
        
class Crawler(object):
    
    db_entries = [
    ('year','INTEGER'),
    ('hdr', 'TEXT'),
    ('path', 'TEXT'),
    ('yrange', 'REAL'),
    ('xrange', 'REAl'),
    ('ypts', 'INTEGER'),
    ('xpts', 'INTEGER'),
    ('ystep', 'REAL'),
    ('xstep', 'REAL'),
    ('angle', 'REAL'),
    ]
    
    def __init__(self,home_dir='/tmp/crawl/'):
        
        self.home = home_dir
        
        self.ptycho_gallery = home_dir+'ptycho_img/'
        self.stxm_gallery = home_dir+'stxm_img/'
        
        if not os.path.exists(self.ptycho_gallery):
            os.makedirs(self.ptycho_gallery)
        
        if not os.path.exists(self.stxm_gallery ):
            os.makedirs(self.stxm_gallery )
        
        self.db_keys = list(dict(self.db_entries).keys())
        
    def extract_from_hdr(self, hdr, render_images=True):
        
        base,fname = os.path.split(hdrf)
        dhdr = eval_hdr(hdrf)
        out = dict.fromkeys(self.db_keys) 
        tm = time.strptime(dhdr['Time'],'%Y %B %d %H:%M:%S')
        out['date'] = time.strftime('%a %b %d %H:%M:%S %Y',tm)
        out['year'] = int(tm.tm_year) 
        out['month'] = int(tm.tm_mon) 
        out['day'] = int(tm.tm_mday) 
        name = fname.replace('.hdr','')
        out['name'] = name
        scan = int(name[-3:])
        out['scan'] = scan

        def write_caption(dct):
            
            caption = """
            {name}
            
            Date:\t{date}
            Path:\t{path}
            Range:\t{xrange}x{yrange} um
            Points:\t{xpts}x{ypts}
            Rotation:\t{angle:.2f}
            """.format(**dct)
            
            return caption
            
        scan_def = dhdr['ScanDefinition']
        if scan_def['Type'] not in ['Image Scan', 'Ptycho Image Scan']:
            return None, None
        out['angle'] = dhdr['AUX11']['LastPosition']
        print(dhdr['ImageScan']['SpatialRegions'])
        spat_regions = dhdr['ImageScan']['SpatialRegions']
        num_reg = spat_regions[0]
        npoints = 0
        for ii in range(1,num_reg+1):
            pars = spat_regions[ii]
            n = pars['YPoints'] * pars['XPoints']
            if n >= npoints:
                # this is assumed to be the actual ptycho scan
                out['yrange'] = pars['YRange']
                out['xrange'] = pars['XRange']
                out['ypts'] = pars['YPoints']
                out['xpts'] = pars['XPoints']
                npoints = n
                out['allpoints'] = n
            else:
                break
                
        energy_reg, pars = dhdr['ImageScan']['EnergyRegions']
        

            
        c = np.array([cmaps.bone(i) for i in range(256)]) * 255 
    
        if render_images:
            if scan_def['Type'] == 'Image Scan':
                f = hdrf.replace('.hdr','_a.xim')
                print("rendering image from " + f)
                A = readASCIIMatrix(f)       
                #im = toimage(A, high=A.max(), low=A.min(), cmin=None, cmax=None, pal=c[:,:3], mode='P')
                img = toimage(A, mode='P')   
                pfile = self.stxm_gallery + fname.replace('.hdr','.png')   
                img.save(pfile )
                out['image'] = pfile
                
            elif scan_def['Type'] == 'Ptycho Image Scan':
                
                cxi_search = base + '/' + name[3:] + '/%03d/' + fname.replace('.hdr','.cxi')
                for i in range(6):
                    fcxi = cxi_search % i
                    if os.path.exists(fcxi):
                        data = get_image_data_from_cxi(fcxi)
                        if data is None:
                            # Case of not reconstructed data
                            break
                        A = np.abs(data)
                        img = toimage(A, pal=c[:,:3], mode='P')
                        pfile = self.ptycho_gallery + fname.replace('.hdr','.png')
                        img.save(pfile)
                        out['image'] = pfile
                        out['path'] = fcxi
                        with open(pfile.replace('.png','.txt'),'w') as f:
                            f.write(write_caption(out))
                            f.close()
                        break
            else:
                img = None
                out['image'] = None
                        
            
        return out, dhdr

if __name__ == '__main__':
    import datetime
    dt = datetime.date(2017,12,8)

    #base = '/home/benders/cosmic-dtn/'
    base = '/cosmic-dtn/'
    data_path = base + 'groups/cosmic/Data/'
    #hdr_dir =  base + 'groups/cosmic/Data/' + dt.strftime('%Y/%m/%y%m%d/')
    hdr_dir =  base + 'groups/cosmic/Data/' + dt.strftime('%y%m%d/')
    
    #hdr_dir = data_path + '2016/12/*/'
     # get all header files
    hdr_files = sorted( glob.glob(hdr_dir +'*.hdr') )
    print("Analysing %d .hdr files in %s" % (len(hdr_files),hdr_dir))
    img_path = data_path + 'images/'
    engine = sa.create_engine('sqlite:///'+data_path+'ptycho.db') #, echo=True)
    
    Base.metadata.create_all(engine)
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=engine)
    session = Session()

    entry = PtychoHdr()
    for hdrf in hdr_files:
        #info, dhdr = C.extract_from_hdr(hdrf)
        success = entry.extract_from_hdr(hdrf)
        if not success:
            continue
        else:
            png_path = entry.png_from_cxi(img_path)
            print("Image at %s" % png_path)
            try:
                session.add(entry)
                session.commit()
            except sa.exc.IntegrityError:
                print("Entry exists. Rolling back")
                session.rollback()
                entry = PtychoHdr()
                continue
                
            
            
"""
show_stxm=True
show_recon=False
)

out_rst = 'autolog.rst'  
autologdir = base + 'autolog/'

if not os.path.exists(autologdir):
    os.makedirs(autologdir)


rstfile = autologdir + out_rst
rst = open(rstfile,'w')


for hdrf in hdr_files[:200]:

    name = os.path.split(hdrf)[1]
    dhdr = eval_hdr(hdrf)
    
    scan = name.replace('.hdr','')[-3:]
    c = np.array([cmaps.hot(i) for i in range(256)]) * 255 
    
    if dhdr.ScanDefinition.Type == 'Image Scan' and show_stxm:
        rst.write(name + '\n')
        A = readASCIIMatrix(hdrf.replace('.hdr','_a.xim'))
        
        #im = toimage(A, high=A.max(), low=A.min(), cmin=None, cmax=None, pal=c[:,:3], mode='P')
        im = toimage(A, mode='P')
        picname = name.replace('.hdr','.png')
        im.save(autologdir + picname)
        rst.write(_pic(name, A.shape[0]))
        
    if dhdr.ScanDefinition.Type == 'Ptycho Image Scan' and show_recon:
        rst.write(name + '\n')
        thisData = readCXI(base + str(day) + scan + '/001/' + name.replace('.hdr','.cxi'))
        A = thisData.getod()
        im = toimage(A, pal=c[:,:3], mode='P')
        picname = name.replace('.hdr','.png')
        im.save(autologdir + picname)
        rst.write(_pic(name, A.shape[0]))
        
rst.close()
os.system("rst2html %s %s" % (rstfile, rstfile.replace('.rst','.html'))) 
"""
