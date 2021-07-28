
import globus_sdk
import requests
import time
import os

# this is to work on Python2 and Python3 -- you can just use raw_input() or
# input() for your specific version
get_input = getattr(__builtins__, 'raw_input', input)

UUIDS = dict(
NERSC_HPSS = '9cd89cfd-6d04-11e5-ba46-22000b92c6ec',
NERSC_DTN = '9d6d994a-6d04-11e5-ba46-22000b92c6ec',
PHASIS_DTN = '65ed6c76-76e3-11e6-8432-22000b97daec',
NERSC_SHARE = '908a3b62-36a4-11e8-b95e-0ac6873fc732',
)

CLIENT_ID = '4fb1257f-31d8-4a0b-b91b-21d3198ea184'
SHARE_PATH = '/project/projectdirs/als/www/cosmic_share/'
PHASIS_DTN_ROOT = '/cosmic-dtn/groups/cosmic/Data/'

client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
scopes = 'openid email profile urn:globus:auth:scope:transfer.api.globus.org:all'
scopes+= ' urn:globus:auth:scope:auth.globus.org:view_identity_set'
scopes+= ' urn:globus:auth:scope:auth.globus.org:view_identities'
client.oauth2_start_flow(requested_scopes = scopes)

authorize_url = client.oauth2_get_authorize_url()
print('Please go to this URL and login: {0}'.format(authorize_url))


auth_code = get_input(
    'Please enter the code you get after login here: ').strip()
token_response = client.oauth2_exchange_code_for_tokens(auth_code)

globus_auth_data = token_response.by_resource_server['auth.globus.org']
globus_transfer_data = token_response.by_resource_server['transfer.api.globus.org']

# most specifically, you want these tokens as strings
AUTH_TOKEN = globus_auth_data['access_token']
TRANSFER_TOKEN = globus_transfer_data['access_token']

authorizer = globus_sdk.AccessTokenAuthorizer(TRANSFER_TOKEN)
tc = globus_sdk.TransferClient(authorizer=authorizer)

def activate_endpoint(ep_id):
    r = tc.endpoint_autoactivate(ep_id, if_expires_in=3600)
    if r['code'] == 'AutoActivationFailed':
        print('Endpoint({}) Not Active! Error! Source message: {}'.format(ep_id, r['message']))
        print("Endpoint requires manual activation, please open "
              "the following URL in a browser to activate the "
              "endpoint:")
        print("https://www.globus.org/app/endpoints/%s/activate" % ep_id)
        # For python 3.X, use input() instead
        get_input("Press ENTER after activating the endpoint:")
        r = tc.endpoint_autoactivate(ep_id, if_expires_in=3600)
    elif r['code'] == 'AutoActivated.CachedCredential':
        print('Endpoint({}) autoactivated using a cached credential'.format(ep_id))
    elif r['code'] == 'AutoActivated.GlobusOnlineCredential':
        print(('Endpoint({}) autoactivated using a built-in Globus credential.').format(ep_id))
    elif r['code'] == 'AlreadyActivated':
        print('Endpoint({}) already active until at least {}'.format(ep_id, 3600))
        
    
def ls(ep,path='~'):
    rep = tc.operation_ls(ep,path=path)
    print('\n'.join([u['type']+'\t'+u['name'] for u in rep.data['DATA']]))
    
def auto_transfer(monitor_transfer=True):
    rep = requests.get('http://128.55.206.17:60000/transfer/all')
    try:
        dct = rep.json()
    except:
        dct = None
    dest_dct={}
    if not dct:
        print('No transfers requested.\n')
    else:
        print('Received the following transfer requests:\n' + '\n'.join(dct.values()))
        tdata = globus_sdk.TransferData(tc, UUIDS['PHASIS_DTN'], UUIDS['NERSC_DTN'], 
            label = "autotransfer",sync_level="checksum", preserve_timestamp=True)
        for k, cxi_path in dct.items():
            fpath = PHASIS_DTN_ROOT + cxi_path.split('Data/')[1]
            fname = os.path.split(cxi_path)[1]
            tpath = SHARE_PATH + fname
            tdata.add_item(fpath, tpath)
            dest_dct[tpath]=k
            print(fpath,tpath)
            rep = requests.post('http://128.55.206.17:60000/transfer/ordered/%s' % k)
        transfer_result = tc.submit_transfer(tdata)
        task_id = transfer_result['task_id']
        while not tc.task_wait(task_id, timeout=10):
            print("Waiting for {0} to complete".format(task_id))
        for info in tc.task_successful_transfers(task_id):
            k = dest_dct.get(info['destination_path'])
            if k is not None:
                print('Transmitted %s to %s \n' % (k,info['destination_path']))
                rep = requests.post('http://128.55.206.17:60000/transfer/completed/%s' % k)

for uuid in UUIDS.values():
    activate_endpoint(uuid)
        
while True:
    transfered = auto_transfer()
    time.sleep(5.)
